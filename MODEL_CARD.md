# Model Card — Predictive Underwriting Risk Classifier

This card follows the structure popularized by Mitchell et al., *Model Cards for Model Reporting* (2019). All numbers are from the executed notebook ([notebooks/responsible_ai_underwriting.ipynb](notebooks/responsible_ai_underwriting.ipynb)).

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

- **Source:** The public medical-cost insurance dataset (~1,338 records), loaded by default from a public URL; an optional Kaggle path exists.
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

XGBoost and calibrated XGBoost tie on accuracy/precision/recall/F1; the calibrated version was chosen for more reliable probability estimates (highest ROC-AUC, 0.939).

## Fairness analysis

Evaluated with Fairlearn across `sex`, `region`, and `smoker`. "Selection rate" = the share of a group predicted "Bad Risk."

| Attribute | Finding | Status |
|---|---|---|
| **Sex** | Baseline selection rate female 0.398 vs male 0.443 (4.4-pt gap, parity ratio ≈ 0.90); recall near-equal (0.879 / 0.881). After `ThresholdOptimizer` with an equalized-odds constraint: female 0.414 vs male 0.443 (2.9-pt gap, ratio ≈ 0.94). | Reduced, not eliminated. Overall accuracy fell 0.933 → 0.925, F1 0.924 → 0.917. |
| **Region** | Selection rate ranges 0.377 (northwest) to 0.481 (northeast), ~10-pt spread (ratio ≈ 0.78). | Measured, **not mitigated**. |
| **Smoker** | All test-set smokers predicted "Bad Risk" (selection rate 1.00) vs 0.272 for non-smokers. | **Left in place by design** — smoking is a recognized actuarial factor. |

## Ethical considerations and limitations

- **Legal context drives the fairness reading.** Smoker-based differentiation is a widely accepted actuarial rating factor, so the large smoker gap is expected and defensible. Sex-based pricing is restricted or prohibited in many jurisdictions, which is why mitigation targets `sex` — and why the residual 2.9-point sex gap still matters. Region rating varies by jurisdiction and line of business; that disparity is unresolved here and would need a jurisdiction-specific decision before any real use.
- **Fairness/accuracy trade-off** is explicit: mitigating the sex gap cost ~0.7 points of accuracy.
- **Small, non-representative data.** ~1,338 records from one public dataset; no claim of generalization.
- **Synthetic target.** The "Bad Risk" label is a thresholded proxy for cost, not a real underwriting outcome.

## Caveats and next steps

The notebook does not yet include a NIST AI RMF scoring section or production monitoring (drift) reports; those are tracked as TODOs in the [README](README.md). Region-level disparity is measured but not addressed.
