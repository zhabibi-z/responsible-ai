# Model Card — Predictive Underwriting Risk Classifier

This card follows the structure popularized by Mitchell et al., *Model Cards for Model Reporting* (2019). Headline performance numbers are from the executed notebook ([notebooks/responsible_ai_underwriting.ipynb](notebooks/responsible_ai_underwriting.ipynb)); the fairness CIs, leakage-free mitigation, and calibration metrics are reproduced deterministically by [evaluate.py](evaluate.py).

## Model details

- **Developer:** Zia Habibi (portfolio project).
- **Date:** 2026.
- **Type:** Binary classifier. The selected model is a probability-calibrated XGBoost (`CalibratedClassifierCV` wrapping an XGBoost classifier). Three other models were trained for comparison: Logistic Regression, a Decision Tree, and an uncalibrated XGBoost.
- **Input features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region` (numeric features scaled; categoricals one-hot encoded via a `ColumnTransformer`).
- **Output:** A predicted class — "Bad Risk" vs "Good Risk" — and a calibrated probability.
- **Version:** Matches the committed notebook. Library versions are pinned in [requirements.txt](requirements.txt).

## Intended use

- **Primary use:** A demonstration of a responsible-AI workflow (explainability + fairness audit) on an insurance risk-classification task. Built for learning and portfolio purposes.
- **Intended users:** Reviewers and practitioners reading the project.
- **Out of scope:** This model is **not** fit for real underwriting decisions. It is trained on a small public dataset, has not been validated on production data, and has known, unresolved fairness gaps (below). Do not use it to price or deny coverage.

## Training data

- **Source:** The public medical-cost insurance dataset (1,338 records, 1,337 after dropping one duplicate row), loaded by default from a public URL; an optional Kaggle path exists.
- **Target construction:** The continuous `charges` column is binarized into `risk_level` — `charges > $10,000` → "Bad Risk." This $10,000 threshold is a modeling choice, not an industry standard.
- **Splits:** 60% train / 20% validation / 20% test, stratified. The test set holds 268 records.
- **Class imbalance:** Addressed on the training fold only, using SMOTE.

## Evaluation data

The held-out 20% test set (268 records), untouched during training and model selection.

## Quantitative performance (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.884 | 0.856 | 0.904 | 0.879 | 0.938 |
| Decision Tree | 0.925 | 0.973 | 0.864 | 0.915 | 0.905 |
| XGBoost | 0.933 | 0.973 | 0.880 | 0.924 | 0.934 |
| **Calibrated XGBoost (selected)** | **0.933** | **0.973** | **0.880** | **0.924** | **0.939** |

XGBoost and calibrated XGBoost tie on the threshold-0.5 classification metrics; the calibrated version was selected for more reliable probabilities. The choice is backed by evidence on the test set: calibration lowered Expected Calibration Error from 0.048 to 0.027 and the Brier score from 0.0599 to 0.0583 (see [evaluate.py](evaluate.py) and [reports/calibration_reliability.png](reports/calibration_reliability.png)).

## Fairness analysis

Evaluated with Fairlearn across `sex`, `region`, and `smoker` on the test set (268 records). "Selection rate" = the share of a group predicted "Bad Risk." Demographic-parity ratios (DPR) carry percentile bootstrap 95% CIs (2,000 resamples, [evaluate.py](evaluate.py)) because the per-group samples are small.

| Attribute | Baseline (test set) | Statistically robust? |
|---|---|---|
| **Sex** | female 0.398 vs male 0.443 (4.4-pt gap); DPR 0.90, 95% CI **[0.67, 0.99]** | **No** — the CI nearly reaches parity (1.0), so the sex gap is not distinguishable from zero at this sample size. |
| **Region** | 0.377 (northwest) to 0.481 (northeast), ~10-pt spread; DPR 0.78, 95% CI **[0.50, 0.90]** | **Yes** — CI upper bound is below 1.0. The most robust disparity; **not mitigated**. |
| **Smoker** | non-smoker 0.272 vs smoker 1.00; DPR 0.27, 95% CI **[0.22, 0.33]** | **Yes** — large and tight. **Left in place by design** (recognized actuarial factor). |

### Sex mitigation, redone without leakage

The notebook applied Fairlearn's `ThresholdOptimizer` (equalized-odds constraint on `sex`) but **fit it on the test set and evaluated on that same set**, which leaks and produced an apparent improvement (gap 4.4 → 3.7 pt, equalized-odds ratio 0.0 → 0.52). [evaluate.py](evaluate.py) redoes it correctly — fit on the validation fold, evaluated on the held-out test fold — and the mitigation **does not transfer**:

| Metric | Baseline | Mitigated (fit on validation) |
|---|---|---|
| sex selection-rate gap | 4.4 pt | 4.5 pt |
| demographic-parity ratio | 0.90 | 0.90 |
| equalized-odds ratio | 0.00 | 0.00 |
| accuracy / F1 | 0.933 / 0.924 | 0.933 / 0.924 |

This is consistent with the wide sex DPR CI above: there is no robust sex gap for a post-hoc threshold (tuned on ~110 validation records per group) to fix, and it does not generalize. The served model is the unmitigated calibrated XGBoost.

## Ethical considerations and limitations

- **Legal context drives the fairness reading.** Smoker-based differentiation is a widely accepted actuarial rating factor, so the large smoker gap is expected and defensible. Sex-based pricing is restricted or prohibited in many jurisdictions — but here the measured sex gap (4.4 pt) is within sampling noise (its DPR CI reaches parity) and post-hoc mitigation did not improve it out-of-sample. The honest reading is that **`region`, not `sex`, is the robust disparity** in this model, and it is left unaddressed pending a jurisdiction-specific policy decision.
- **Mitigation had no measurable effect.** Once the test-set leakage is removed, `ThresholdOptimizer` on `sex` left accuracy, F1, the parity ratio, and the equalized-odds ratio unchanged. Reporting it as a "fairness/accuracy trade-off" would overstate what it achieved.
- **Small, non-representative data.** ~1,338 records from one public dataset; no claim of generalization. Per-group test samples (e.g. 55 smokers) are small enough that several fairness numbers are dominated by sampling noise — hence the bootstrap CIs.
- **Synthetic target.** The "Bad Risk" label is a thresholded proxy for cost, not a real underwriting outcome.

## Caveats and next steps

- The notebook now includes a NIST AI RMF mapping (Section 7), Evidently data-summary and drift reports (Section 8), and a Gradio interface backed by the saved model with an out-of-distribution check (Section 9). The standalone `monitoring.py` regenerates the drift reports from the same dataset.
- Region-level disparity is measured but not addressed; a region policy decision is needed before any real use.
- The served model is the unmitigated calibrated XGBoost; the sex mitigation exists only as a separate analysis step and, evaluated without leakage, did not improve fairness on the test set.
- No privacy or access controls are in place. Add authentication and a PII policy before using real applicant data.
