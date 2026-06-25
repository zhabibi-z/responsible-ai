# Responsible AI for Predictive Underwriting

A notebook that builds an insurance risk-classification model and then audits it for transparency and fairness, rather than stopping at accuracy.

## Problem

Predictive underwriting uses a model to estimate an applicant's risk and price or screen policies accordingly. Because those decisions affect access and cost for real people, the model needs more than good accuracy: it has to be explainable to underwriters and regulators, and it has to be checked for disparate treatment across groups such as sex and region. This project treats that checking as part of the deliverable, not an afterthought.

The data is the public medical-cost insurance dataset (1,338 records, 1,337 after dropping one duplicate row; columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`). The continuous `charges` field is converted into a binary target `risk_level`, where charges above $10,000 are labeled "Bad Risk." The model predicts that label.

## What this project does

The notebook runs as an ordered pipeline:

1. **Data ingestion** — loads `insurance.csv` from a public URL by default, with an optional Kaggle path. A 60/20/20 train/validation/test split is used.
2. **EDA and preprocessing** — schema validation, a `ColumnTransformer` for scaling and one-hot encoding, and SMOTE to address class imbalance in the training fold.
3. **Four models** — Logistic Regression and a Decision Tree as interpretable baselines, XGBoost as the performance benchmark, and a probability-calibrated XGBoost.
4. **Explainability** — SHAP (global beeswarm and local waterfall/dependence plots), LIME for local model-agnostic explanations, and direct inspection of the Logistic Regression coefficients.
5. **Test-set evaluation** — all four models re-scored on the held-out test set.
6. **Fairness audit (Fairlearn)** — per-group metrics across `sex`, `region`, and `smoker`, followed by a `ThresholdOptimizer` mitigation pass on `sex`.

### Project Completion Status

All major sections have been implemented:

- [x] **Model card** — see [MODEL_CARD.md](MODEL_CARD.md)
- [x] **NIST AI RMF scoring section** — Section 7 in notebook with maturity assessment across core functions and seven trustworthiness characteristics
- [x] **Monitoring infrastructure** — Evidently AI integration in Section 8; standalone monitoring script: `python monitoring.py`
- [x] **System design & Gradio app** — Section 9 in notebook; interactive application: `python app.py`

## Key results

### Best model (held-out test set)

The calibrated XGBoost model was selected. XGBoost and calibrated XGBoost tie on the headline classification metrics; calibration was preferred for more reliable probability estimates — and the evidence backs it: on the test set, calibration cut Expected Calibration Error from 0.048 to 0.027 and the Brier score from 0.0599 to 0.0583 (reliability diagram in [reports/calibration_reliability.png](reports/calibration_reliability.png), reproduced by [evaluate.py](evaluate.py)). Test-set numbers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.884 | 0.856 | 0.904 | 0.879 | 0.938 |
| Decision Tree | 0.925 | 0.973 | 0.864 | 0.915 | 0.905 |
| XGBoost | 0.933 | 0.973 | 0.880 | 0.924 | 0.934 |
| **Calibrated XGBoost** | **0.933** | **0.973** | **0.880** | **0.924** | **0.939** |

Read plainly: the chosen model gets about 93% of test cases right, and when it labels an applicant "Bad Risk" it is correct about 97% of the time (precision), while catching 88% of the actual bad-risk cases (recall).

### Fairness findings (calibrated XGBoost, test set)

Selection rate = the share of a group predicted "Bad Risk." Because the per-group
samples are small (e.g. 55 smokers), demographic-parity ratios (DPR) are reported
with percentile bootstrap 95% CIs from [evaluate.py](evaluate.py) — the CI, not the
point estimate, is what tells you whether a gap is real.

| Attribute | Baseline selection rate | DPR (95% CI) | Robust? |
|---|---|---|---|
| **Sex** | female 0.398 / male 0.443 (4.4-pt gap) | 0.90 **[0.67, 0.99]** | **No** — CI reaches parity |
| **Region** | 0.377 (NW) → 0.481 (NE), ~10-pt spread | 0.78 **[0.50, 0.90]** | **Yes** — CI below 1.0 |
| **Smoker** | non-smoker 0.272 / smoker 1.00 | 0.27 **[0.22, 0.33]** | **Yes** — large, by design |

**Sex mitigation does not transfer (and the original "win" was leakage).** The
notebook fit Fairlearn's `ThresholdOptimizer` on the *test* set and scored it on
that same set, which leaks and reported a gap reduction 4.4 → 3.7 pt (equalized-odds
ratio 0.0 → 0.52). Redone correctly in [evaluate.py](evaluate.py) — fit on the
**validation** fold, evaluated on the held-out **test** fold — the mitigation leaves
everything essentially unchanged: gap 4.4 → 4.5 pt, DPR 0.90 → 0.90, equalized-odds
ratio 0.0 → 0.0, accuracy and F1 flat at 0.933 / 0.924. This is exactly what the wide
sex DPR CI predicts: there is no statistically robust sex gap for a post-hoc threshold
(tuned on ~110 validation records per group) to fix.

Bottom line: **`region` is the one robust disparity** (and it is measured but not
mitigated, pending a jurisdiction policy decision); the **`smoker`** gap is large,
robust, and intentional; and the **`sex`** gap is within sampling noise, so the honest
conclusion is "not distinguishable from parity here," not "reduced by mitigation." The
audit surfaces this rather than claiming the model was made "fair."

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, LIME, Fairlearn, matplotlib, seaborn, joblib, and the Kaggle API (optional data path). Pinned versions are in [requirements.txt](requirements.txt).

## Repository structure

```
responsible-ai/
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py                  # Local Gradio launcher
├── underwriting_demo.py    # Shared UI + inference logic (used by both launchers)
├── monitoring.py           # Standalone Evidently monitoring script
├── notebooks/
│   ├── responsible_ai_underwriting.ipynb
│   └── models/             # Trained model + preprocessor (committed)
├── hf_space/               # Self-contained Hugging Face Space
│   ├── app.py              # Space launcher (imports underwriting_demo.py)
│   ├── README.md
│   ├── requirements.txt
│   └── models/             # Copy of the artifacts the Space serves
└── reports/                # Evidently HTML reports written by the notebook and monitoring.py
```

The notebook is committed with its executed outputs (charts, tables) intact. The trained model and preprocessor are committed under `notebooks/models/` (and `hf_space/models/`) so `app.py` and the Hugging Face Space run without re-executing the notebook. Running the notebook or `monitoring.py` writes the Evidently HTML reports into `reports/`.

## How to run

### Main Notebook (Complete Analysis)
```bash
git clone https://github.com/zhabibi-z/responsible-ai.git
cd responsible-ai
pip install -r requirements.txt
jupyter notebook notebooks/responsible_ai_underwriting.ipynb
```

The notebook includes:
- Sections 1-6: Core ML pipeline (data → training → fairness audit)
- **Section 7:** NIST AI RMF evaluation with maturity scores
- **Section 8:** Evidently AI monitoring setup
- **Section 9:** System design & Gradio app code
- Section 11: Executive summary & recommendations

### Generate Monitoring Reports
```bash
python monitoring.py
```
Reads the same insurance dataset, reproduces the train/test feature split, and writes Evidently reports to `reports/`:
- `monitoring_data_summary.html` — column statistics for train vs test
- `monitoring_data_drift.html` — feature drift, test set vs training reference (baseline)
- `monitoring_data_drift_alarm.html` — drift under a shifted (ageing, higher-BMI) test population

The notebook's Section 8 writes the same kind of reports under the names `data_summary_report.html`, `data_drift_report.html`, and `data_drift_alarm_report.html`. Open any HTML file in a web browser to view detailed findings.

### Reproduce the fairness & calibration evaluation
```bash
python evaluate.py
```
A deterministic, no-retraining script that loads the committed artifacts,
reproduces the exact 60/20/20 split, and prints the bootstrap-CI fairness
tables, the leakage-free mitigation result, and the calibration metrics. It
writes `reports/evaluation_results.json` and `reports/calibration_reliability.png`.
Run the smoke tests with `pytest -q`.

### Launch Interactive Application
```bash
python app.py
```
Opens Gradio interface at `http://localhost:7860` with:
- **Make Prediction** — Input applicant features, get risk score + fairness context
- **Model Information** — Performance metrics, fairness analysis, limitations
- **Feature Guide** — Interpretation of each input feature
- **System Architecture** — Technical design, deployment, governance

The default configuration uses `DATA_SOURCE="URL"`, which pulls the dataset from a public URL and needs **no Kaggle credentials**. To use the Kaggle source instead, set `DATA_SOURCE="KAGGLE"` and provide `KAGGLE_USERNAME` / `KAGGLE_KEY` (the notebook reads them from Colab secrets, not from a committed file).

## Responsible AI framing

The work is organized around the NIST AI Risk Management Framework's idea of building governance into the model lifecycle — explainability and a fairness audit are run as named pipeline stages, not bolted on at the end. Section 7 of the notebook scores the project against the RMF core functions and the seven trustworthiness characteristics, with each justification pointing to a section that ran and a gap plus remediation for each.

One legal nuance drives the fairness reading. `smoker` status is a legitimate, widely accepted actuarial rating factor, so the large smoker/non-smoker gap is expected and defensible. Sex-based pricing, by contrast, is restricted or prohibited in many jurisdictions, which is why the mitigation pass targets `sex` specifically — but the measured sex gap turns out to be within sampling noise (its DPR confidence interval reaches parity), and a leakage-free `ThresholdOptimizer` pass does not improve it out-of-sample, so the model is not "made fair" on `sex` so much as shown to have no robust sex gap at this sample size. Geographic (`region`) rating is permitted in some lines and restricted in others; that disparity is the statistically robust one here, measured but not resolved, and would need a jurisdiction-specific decision before deployment.

## License

MIT — see [LICENSE](LICENSE).
