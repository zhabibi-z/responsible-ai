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

### Not yet built

The notebook's closing summary references a NIST AI RMF scoring section, a model card, production monitoring, and a system-design write-up. Those sections are **not present in the executed notebook** (it goes from the fairness audit straight to the executive summary). `evidently` and `gradio` appear in the install cell but are never imported or run, so there are no monitoring reports or app in this repo yet.

- [x] Model card — see [MODEL_CARD.md](MODEL_CARD.md)
- [ ] TODO: NIST AI RMF scoring section
- [ ] TODO: monitoring (Evidently) reports in `reports/`
- [ ] TODO: system-design section / Gradio app

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
- After `ThresholdOptimizer` (equalized-odds constraint on `sex`): female 0.414 vs male 0.443 → gap narrows to 2.9 points (ratio ≈ 0.94). The gap shrank but did not close. Overall accuracy fell from 0.933 to 0.925 and F1 from 0.924 to 0.917 — the usual fairness/accuracy trade-off.

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
├── notebooks/
│   └── responsible_ai_underwriting.ipynb
└── reports/                # for monitoring HTML output (none generated yet)
```

The notebook is committed with its executed outputs (charts, tables) intact.

## How to run

```bash
git clone https://github.com/zhabibi-z/responsible-ai.git
cd responsible-ai
pip install -r requirements.txt
jupyter notebook notebooks/responsible_ai_underwriting.ipynb
```

The default configuration uses `DATA_SOURCE="URL"`, which pulls the dataset from a public URL and needs **no Kaggle credentials**. To use the Kaggle source instead, set `DATA_SOURCE="KAGGLE"` and provide `KAGGLE_USERNAME` / `KAGGLE_KEY` (the notebook reads them from Colab secrets, not from a committed file).

## Responsible AI framing

The work is organized around the NIST AI Risk Management Framework's idea of building governance into the model lifecycle — explainability and a fairness audit are run as named pipeline stages, not bolted on at the end. (A formal section that scores the project against the RMF functions is listed above as a TODO.)

One legal nuance drives the fairness reading. `smoker` status is a legitimate, widely accepted actuarial rating factor, so the large smoker/non-smoker gap is expected and defensible. Sex-based pricing, by contrast, is restricted or prohibited in many jurisdictions, which is why the mitigation pass targets `sex` specifically — and why the residual sex gap matters even though it is small. Geographic (`region`) rating is permitted in some lines and restricted in others; that disparity is measured here but not resolved, and would need a jurisdiction-specific decision before deployment.

## License

MIT — see [LICENSE](LICENSE).
