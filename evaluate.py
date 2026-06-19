#!/usr/bin/env python
"""
Reproducible fairness and calibration evaluation for the underwriting model.

This is a deterministic, scripted entry point that does NOT retrain anything: it
loads the committed calibrated XGBoost model, the raw XGBoost model, and the
preprocessor, reproduces the notebook's exact 60/20/20 split, and reports:

  1. Baseline group fairness (sex / region / smoker) on the test set, each
     selection rate and demographic-parity ratio with a bootstrap 95% CI, so
     the gaps are reported with uncertainty rather than as bare point estimates.

  2. Fairness mitigation done correctly. The notebook originally fit Fairlearn's
     ThresholdOptimizer on the *test* set and then evaluated on that same test
     set, which leaks and overstates the benefit. Here the optimizer is fit on
     the held-out VALIDATION fold and evaluated on the untouched TEST fold.

  3. Calibration quality: Brier score and Expected Calibration Error (ECE) for
     the raw vs. calibrated XGBoost on the test set, plus a reliability diagram
     written to reports/. This is the evidence behind choosing the calibrated
     model "for better probabilities".

Run:
    python evaluate.py

Outputs:
    - Structured report to stdout
    - reports/evaluation_results.json
    - reports/calibration_reliability.png
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write the figure without a display
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, brier_score_loss, recall_score,
)
from sklearn.calibration import calibration_curve
from fairlearn.metrics import (
    MetricFrame, selection_rate, demographic_parity_ratio, equalized_odds_ratio,
)
from fairlearn.postprocessing import ThresholdOptimizer

# --- Config (mirrors the notebook's CONFIG and monitoring.py) ---------------
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)
RISK_THRESHOLD = 10000.0
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 0.25 * 0.8 = 0.2 of the whole -> 60/20/20 train/val/test
N_BOOTSTRAP = 2000
N_ECE_BINS = 10
DECISION_THRESHOLD = 0.5

NUMERICAL = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
FEATURES = NUMERICAL + CATEGORICAL

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "notebooks", "models")
REPORTS_DIR = os.path.join(_HERE, "reports")


# --- Data --------------------------------------------------------------------
def load_splits():
    """Reproduce the notebook's train / val / test feature frames from the CSV."""
    df = pd.read_csv(DATA_URL)
    # The notebook drops the dataset's single duplicate row (1338 -> 1337)
    # before splitting; reproduce that exactly so this split matches the one the
    # committed model was trained on.
    df = df.drop_duplicates().reset_index(drop=True)
    df["risk_level"] = (df["charges"] > RISK_THRESHOLD).astype(int)
    X = df[FEATURES]
    y = df["risk_level"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE,
        random_state=RANDOM_SEED, stratify=y_train_val,
    )
    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True), y_val.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True),
    )


# --- Bootstrap helpers -------------------------------------------------------
def bootstrap_ci(stat_fn, n, rng, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap CI for a statistic computed over n paired rows.

    stat_fn receives an index array (positions into the test arrays) and returns
    a scalar; NaN returns (e.g. an empty resampled subgroup) are dropped.
    """
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat_fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def selection_rate_for(y_pred, mask):
    sub = y_pred[mask]
    return float(sub.mean()) if sub.size else float("nan")


def dp_ratio(y_pred, groups):
    """Demographic-parity ratio = min group selection rate / max group rate."""
    rates = [y_pred[groups == g].mean() for g in np.unique(groups) if (groups == g).any()]
    rates = [r for r in rates if not np.isnan(r)]
    if not rates or max(rates) == 0:
        return float("nan")
    return float(min(rates) / max(rates))


# --- Calibration helpers -----------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=N_ECE_BINS):
    """Equal-width-binning ECE: sum_b (n_b/N) * |acc_b - conf_b|."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        conf = y_prob[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # Load artifacts (no retraining).
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    calibrated = joblib.load(os.path.join(MODELS_DIR, "calibrated_xgboost_model.joblib"))
    raw_xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    y_test_np = y_test.to_numpy()

    # Calibrated model scores on the test set.
    proba_cal = calibrated.predict_proba(X_test_t)[:, 1]
    y_pred_cal = (proba_cal > DECISION_THRESHOLD).astype(int)
    # Raw XGBoost (DataFrame to keep feature names; fall back to ndarray).
    try:
        proba_raw = raw_xgb.predict_proba(X_test_t)[:, 1]
    except Exception:
        proba_raw = raw_xgb.predict_proba(np.asarray(X_test_t))[:, 1]

    results = {"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)}

    # ---- 1. Baseline fairness with bootstrap CIs ----------------------------
    print("=" * 78)
    print("1. BASELINE GROUP FAIRNESS (calibrated XGBoost, test set, threshold 0.5)")
    print(f"   test n = {len(y_test)}   (overall selection rate "
          f"{y_pred_cal.mean():.3f})")
    print("=" * 78)

    fairness = {}
    for attr in ["sex", "region", "smoker"]:
        groups = X_test[attr].to_numpy()
        print(f"\n  [{attr}]")
        attr_block = {"groups": {}}
        for g in sorted(np.unique(groups)):
            mask = groups == g
            sr = selection_rate_for(y_pred_cal, mask)
            lo, hi = bootstrap_ci(
                lambda idx, m=attr, gg=g: selection_rate_for(
                    y_pred_cal[idx], X_test[m].to_numpy()[idx] == gg),
                n=len(y_test), rng=rng,
            )
            n_g = int(mask.sum())
            print(f"    {g:<10} selection_rate {sr:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]  (n={n_g})")
            attr_block["groups"][str(g)] = {
                "selection_rate": sr, "ci95": [lo, hi], "n": n_g}

        dpr = demographic_parity_ratio(
            y_true=y_test_np, y_pred=y_pred_cal, sensitive_features=groups)
        dlo, dhi = bootstrap_ci(
            lambda idx, m=attr: dp_ratio(y_pred_cal[idx], X_test[m].to_numpy()[idx]),
            n=len(y_test), rng=rng,
        )
        print(f"    demographic-parity ratio {dpr:.3f}  95% CI [{dlo:.3f}, {dhi:.3f}]")
        attr_block["demographic_parity_ratio"] = float(dpr)
        attr_block["demographic_parity_ratio_ci95"] = [dlo, dhi]
        fairness[attr] = attr_block
    results["baseline_fairness"] = fairness

    acc = accuracy_score(y_test_np, y_pred_cal)
    alo, ahi = bootstrap_ci(
        lambda idx: accuracy_score(y_test_np[idx], y_pred_cal[idx]),
        n=len(y_test), rng=rng)
    print(f"\n  overall accuracy {acc:.3f}  95% CI [{alo:.3f}, {ahi:.3f}]")
    results["overall_accuracy"] = {"value": float(acc), "ci95": [alo, ahi]}

    # ---- 2. Mitigation fit on VALIDATION, evaluated on TEST -----------------
    print("\n" + "=" * 78)
    print("2. FAIRNESS MITIGATION (ThresholdOptimizer on 'sex')")
    print("   FIT on the validation fold, EVALUATED on the held-out test fold")
    print("=" * 78)

    thr = ThresholdOptimizer(
        estimator=calibrated,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    thr.fit(X_val_t, y_val.to_numpy(), sensitive_features=X_val["sex"].to_numpy())
    y_pred_mit = thr.predict(
        X_test_t, sensitive_features=X_test["sex"].to_numpy(),
        random_state=RANDOM_SEED,  # deterministic interpolated thresholds
    )

    sex_test = X_test["sex"].to_numpy()
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "recall": recall_score},
        y_true=y_test_np, y_pred=y_pred_mit, sensitive_features=sex_test,
    )
    base_sr = {g: selection_rate_for(y_pred_cal, sex_test == g) for g in ["female", "male"]}
    mit_sr = mf.by_group["selection_rate"].to_dict()

    base_gap = abs(base_sr["female"] - base_sr["male"]) * 100
    mit_gap = abs(mit_sr["female"] - mit_sr["male"]) * 100
    dpr_base = demographic_parity_ratio(y_true=y_test_np, y_pred=y_pred_cal, sensitive_features=sex_test)
    dpr_mit = demographic_parity_ratio(y_true=y_test_np, y_pred=y_pred_mit, sensitive_features=sex_test)
    eo_base = equalized_odds_ratio(y_true=y_test_np, y_pred=y_pred_cal, sensitive_features=sex_test)
    eo_mit = equalized_odds_ratio(y_true=y_test_np, y_pred=y_pred_mit, sensitive_features=sex_test)
    acc_mit = accuracy_score(y_test_np, y_pred_mit)
    f1_base = f1_score(y_test_np, y_pred_cal)
    f1_mit = f1_score(y_test_np, y_pred_mit)

    print(f"\n  selection rate  female {base_sr['female']:.3f} -> {mit_sr['female']:.3f}")
    print(f"  selection rate  male   {base_sr['male']:.3f} -> {mit_sr['male']:.3f}")
    print(f"  sex gap         {base_gap:.1f}pt -> {mit_gap:.1f}pt")
    print(f"  dem.-parity ratio {dpr_base:.3f} -> {dpr_mit:.3f}")
    print(f"  equalized-odds ratio {eo_base:.3f} -> {eo_mit:.3f}")
    print(f"  accuracy        {acc:.3f} -> {acc_mit:.3f}")
    print(f"  F1              {f1_base:.3f} -> {f1_mit:.3f}")

    results["mitigation_sex"] = {
        "fit_on": "validation", "evaluated_on": "test",
        "selection_rate": {
            "female": [float(base_sr["female"]), float(mit_sr["female"])],
            "male": [float(base_sr["male"]), float(mit_sr["male"])],
        },
        "sex_gap_pt": [float(base_gap), float(mit_gap)],
        "demographic_parity_ratio": [float(dpr_base), float(dpr_mit)],
        "equalized_odds_ratio": [float(eo_base), float(eo_mit)],
        "accuracy": [float(acc), float(acc_mit)],
        "f1": [float(f1_base), float(f1_mit)],
    }

    # ---- 3. Calibration: raw vs calibrated ----------------------------------
    print("\n" + "=" * 78)
    print("3. CALIBRATION QUALITY (test set)")
    print("=" * 78)
    brier_raw = brier_score_loss(y_test_np, proba_raw)
    brier_cal = brier_score_loss(y_test_np, proba_cal)
    ece_raw = expected_calibration_error(y_test_np, proba_raw)
    ece_cal = expected_calibration_error(y_test_np, proba_cal)
    print(f"\n  Brier score   raw {brier_raw:.4f}  ->  calibrated {brier_cal:.4f}"
          f"   (lower is better)")
    print(f"  ECE           raw {ece_raw:.4f}  ->  calibrated {ece_cal:.4f}"
          f"   (lower is better)")
    results["calibration"] = {
        "brier": {"raw": float(brier_raw), "calibrated": float(brier_cal)},
        "ece": {"raw": float(ece_raw), "calibrated": float(ece_cal)},
    }

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for proba, name in [(proba_raw, "Raw XGBoost"), (proba_cal, "Calibrated XGBoost")]:
        frac_pos, mean_pred = calibration_curve(y_test_np, proba, n_bins=N_ECE_BINS, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label=name)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Fraction of positives (bin)")
    ax.set_title("Reliability diagram — raw vs calibrated XGBoost (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "calibration_reliability.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved reliability diagram: {plot_path}")

    out_path = os.path.join(REPORTS_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results JSON:        {out_path}")
    return results


if __name__ == "__main__":
    main()
