# Responsible AI for Predictive Underwriting

A notebook that builds an insurance risk-classification model and then audits it for transparency and fairness, rather than stopping at accuracy.

## Problem

Predictive underwriting uses a model to estimate an applicant's risk and price or screen policies accordingly. Because those decisions affect access and cost for real people, the model needs more than good accuracy: it has to be explainable to underwriters and regulators, and it has to be checked for disparate treatment across groups such as sex and region. This project treats that checking as part of the deliverable, not an afterthought.

The data is the public medical-cost insurance dataset (~1,338 records; columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`). The continuous `charges` field is converted into a binary target `risk_level`, where charges above $10,000 are labeled "Bad Risk." The model predicts that label.

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

The calibrated XGBoost model was selected. XGBoost and calibrated XGBoost tie on the headline classification metrics; calibration was preferred for more reliable probability estimates. Test-set numbers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.884 | 0.856 | 0.904 | 0.879 | 0.938 |
| Decision Tree | 0.925 | 0.973 | 0.864 | 0.915 | 0.905 |
| XGBoost | 0.933 | 0.973 | 0.880 | 0.924 | 0.934 |
| **Calibrated XGBoost** | **0.933** | **0.973** | **0.880** | **0.924** | **0.939** |

Read plainly: the chosen model gets about 93% of test cases right, and when it labels an applicant "Bad Risk" it is correct about 97% of the time (precision), while catching 88% of the actual bad-risk cases (recall).

### Fairness findings (Section 6, calibrated XGBoost, test set)

These are read from the per-group selection-rate tables in the notebook. Selection rate = the share of a group predicted "Bad Risk."

**Sex — small gap, partially reduced.**
- Baseline selection rate: female 0.398 vs male 0.443 → a 4.4-point gap (demographic-parity ratio ≈ 0.90).
- Recall was nearly equal across sex at baseline (female 0.879, male 0.881).
- After `ThresholdOptimizer` (equalized-odds constraint on `sex`): female 0.398 vs male 0.436 → gap narrows to 3.7 points (demographic-parity ratio ≈ 0.91; equalized-odds ratio 0.0 → 0.52). The gap shrank but did not close. Overall accuracy fell from 0.933 to 0.929 and F1 from 0.924 to 0.920 — the usual fairness/accuracy trade-off. The mitigation is an analysis step and is not wired into the served model.

**Region — not mitigated.** Baseline selection rates ranged from 0.377 (northwest) to 0.481 (northeast), about a 10-point spread (ratio ≈ 0.78). No mitigation was applied to `region`, so this disparity remains in the final model.

**Smoker — large gap, left in place by design.** Every smoker in the test set was predicted "Bad Risk" (selection rate 1.00) versus 0.272 for non-smokers. This is the strongest signal in the data and is left unmitigated: smoking is a recognized actuarial risk factor (see the framing note below).

Bottom line: bias was reduced on `sex` but not eliminated, `region` disparity was measured but not addressed, and the `smoker` gap is intentional and defensible. The audit surfaces these honestly rather than claiming the model is "fair."

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

One legal nuance drives the fairness reading. `smoker` status is a legitimate, widely accepted actuarial rating factor, so the large smoker/non-smoker gap is expected and defensible. Sex-based pricing, by contrast, is restricted or prohibited in many jurisdictions, which is why the mitigation pass targets `sex` specifically — and why the residual sex gap matters even though it is small. Geographic (`region`) rating is permitted in some lines and restricted in others; that disparity is measured here but not resolved, and would need a jurisdiction-specific decision before deployment.

## License

MIT — see [LICENSE](LICENSE).
