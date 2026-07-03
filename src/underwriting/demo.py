#!/usr/bin/env python
# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
Shared Gradio UI and inference logic for the underwriting demo.

The repo-root app.py imports from this module for both local runs and the
Hugging Face Space. It resolves the model path and launches the server;
everything else -- the prediction path, the fairness context, and the
documentation tabs -- lives here.
"""

import os
import json

import joblib
import pandas as pd
import gradio as gr

# Feature order expected by the saved preprocessor.
NUMERICAL_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURE_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_model_artifacts(candidate_dirs):
    """Load the calibrated XGBoost model and preprocessor.

    Searches each directory in ``candidate_dirs`` in order and returns the
    first one that holds both artifacts. Raises FileNotFoundError if none do.
    """
    for base in candidate_dirs:
        model_path = os.path.join(base, "calibrated_xgboost_model.joblib")
        preproc_path = os.path.join(base, "preprocessor.joblib")
        if os.path.exists(model_path) and os.path.exists(preproc_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preproc_path)
            print(f"Loaded model: {model_path}")
            print(f"Loaded preprocessor: {preproc_path}")
            return model, preprocessor

    searched = "\n  ".join(candidate_dirs)
    raise FileNotFoundError(
        "Could not find calibrated_xgboost_model.joblib and preprocessor.joblib. "
        "Run the notebook (notebooks/responsible_ai_underwriting.ipynb) first; it "
        "saves them under notebooks/models/. Searched:\n  " + searched
    )


def make_predict_fn(model, preprocessor):
    """Build the prediction callback bound to a loaded model and preprocessor."""

    def predict_risk(age, sex, bmi, children, smoker, region):
        try:
            # Single-row frame in the preprocessor's expected order, then score
            # with the calibrated XGBoost model.
            row = pd.DataFrame([{
                "age": age, "bmi": bmi, "children": children,
                "sex": sex, "smoker": smoker, "region": region,
            }])[FEATURE_ORDER]
            X = preprocessor.transform(row)

            probability = float(model.predict_proba(X)[0, 1])
            prediction = int(model.predict(X)[0])
            risk_class = "Bad Risk" if prediction == 1 else "Good Risk"

            result = {
                "Prediction": risk_class,
                "Probability (Bad Risk)": f"{probability * 100:.1f}%",
                "Probability (Good Risk)": f"{(1 - probability) * 100:.1f}%",
            }

            notes = []
            if sex == "female":
                notes.append(
                    "Sex: the baseline selection-rate gap was 4.4pt (women 39.8% vs "
                    "men 44.3%), but the demographic-parity ratio CI [0.67, 0.99] "
                    "reaches parity, so the gap is not statistically robust at this "
                    "sample size, and a leakage-free ThresholdOptimizer pass did not "
                    "improve it. Sex-based pricing is restricted in many jurisdictions, "
                    "so it is worth monitoring as data grows."
                )
            if region == "northeast":
                notes.append(
                    "Region: Northeast has the highest baseline selection rate (48.1%). "
                    "Regional disparities are measured but not mitigated, pending a "
                    "jurisdiction policy decision."
                )
            elif region == "northwest":
                notes.append(
                    "Region: Northwest has the lowest baseline selection rate (37.7%)."
                )
            if smoker == "yes":
                notes.append(
                    "Smoker: near-total selection rate. This gap is intentional and "
                    "defensible -- smoking is a recognized actuarial factor -- so no "
                    "mitigation was applied."
                )
            else:
                notes.append(
                    "Non-smoker: lower baseline selection rate (27.2%)."
                )

            if notes:
                result["Fairness & Context"] = "\n\n".join(notes)

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error: {e}"

    return predict_risk


def get_model_info():
    """Return the Model Information tab (markdown)."""
    return """
# Model Information & Performance

## Selected Model
- **Type:** Calibrated XGBoost (`CalibratedClassifierCV` wrapping XGBoost)
- **Rationale:** Better-calibrated probabilities than the uncalibrated models,
  at no cost to the classification metrics (see the calibration table below)
- **Version:** 1.0

## Test Set Performance
All metrics are on the held-out test set (268 records, the 20% split).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 88.4% | 85.6% | 90.4% | 87.9% | 0.938 |
| Decision Tree | 92.5% | 97.3% | 86.4% | 91.5% | 0.905 |
| XGBoost (uncalibrated) | 93.3% | 97.3% | 88.0% | 92.4% | 0.934 |
| **Calibrated XGBoost** | **93.3%** | **97.3%** | **88.0%** | **92.4%** | **0.939** |

## Pipeline
- **Features:** age, sex, bmi, children, smoker, region
- **Preprocessing:** `ColumnTransformer` (standardise numerics, one-hot categoricals)
- **Imbalance:** SMOTE on the training fold only
- **Output:** binary class + calibrated probability

## Calibration (why the calibrated model was chosen)
Isotonic calibration improved probability quality without changing the
threshold-0.5 classification metrics:

| Metric | Raw XGBoost | Calibrated |
|--------|-------------|------------|
| Expected Calibration Error | 0.048 | **0.027** |
| Brier score | 0.0599 | **0.0583** |

## Fairness
Per-group samples are small, so demographic-parity ratios (DPR) are reported with
bootstrap 95% confidence intervals; the interval, not the point estimate, is what
says whether a gap is real.

- **Sex** — DPR 0.90, 95% CI [0.67, 0.99]. The interval reaches parity, so the
  4.4pt baseline gap is not distinguishable from zero at this sample size. A
  leakage-free `ThresholdOptimizer` pass did not change it out-of-sample.
- **Region** — DPR 0.78, 95% CI [0.50, 0.90]. The upper bound is below 1.0, so
  this is the one statistically robust disparity. Measured but not mitigated,
  pending a jurisdiction policy decision.
- **Smoker** — DPR 0.27, 95% CI [0.22, 0.33]. A large, robust gap, left in place
  by design: smoking is a recognized actuarial rating factor.

## Known Limitations
- Small public dataset (~1,300 records); no validation on production data.
- Synthetic target: "Bad Risk" is a $10,000-charges threshold, not a real outcome.
- One robust, unmitigated fairness gap (region).
- Legal context is jurisdiction-specific and only sketched here.

## Intended Use
A learning and portfolio demonstration of a responsible-AI workflow, **not** a
system for real underwriting decisions. See `MODEL_CARD.md` for the full intended-
use and out-of-scope statement.
"""


def get_feature_guide():
    """Return the Feature Guide tab (markdown)."""
    return """
# Feature Guide

How to read each model input, and the prediction it produces.

### Age (18-70)
Strong predictor of healthcare cost; risk rises with age and accelerates later in
life.

### Sex (male / female)
A proxy for cost differences. Sex-based pricing is restricted in many
jurisdictions, so it is treated as a sensitive attribute. The baseline
female/male selection-rate gap (4.4pt) is within sampling noise (DPR CI
[0.67, 0.99] reaches parity) and post-hoc mitigation did not improve it, so the
honest reading is "no robust sex gap at this sample size," not "bias removed."

### BMI (12-55 kg/m^2)
Higher BMI correlates with higher healthcare cost. WHO bands: normal 18.5-24.9,
overweight 25-29.9, obese >= 30.

### Children (0-5)
Number of dependents; a weak signal in this model.

### Smoker (yes / no)
The dominant signal (top feature by SHAP importance): smokers are predicted "Bad
Risk" at a near-100% rate vs 27.2% for non-smokers. The gap is large but
intentional and defensible -- smoking is a recognized actuarial rating factor --
so no mitigation was applied.

### Region (northeast / northwest / southeast / southwest)
Baseline selection rates span ~10 points (northwest 37.7% to northeast 48.1%,
DPR 0.78). This is the one statistically robust disparity; it is measured but not
mitigated, pending a jurisdiction policy decision.

## Reading the prediction
- **Prediction:** "Bad Risk" = predicted annual charges above $10,000; "Good Risk"
  = below.
- **Probability (Bad / Good Risk):** calibrated class probabilities that sum to
  100%. Calibration makes them usable for pricing or threshold decisions, not just
  ranking.
- **Fairness & Context:** per-group notes for the sensitive attributes, drawn from
  the audit.

The model is decision-support, not an automated decision-maker: it should inform
an underwriter's judgment, never replace it, and should not be the sole basis for
a denial.
"""


def get_system_design():
    """Return the System Design tab (markdown)."""
    return """
# System Design

This is a single-machine, reproducible project, **not** a deployed service. Every
stage below runs from the notebook or the scripts in `src/underwriting/`.

```
Data (public insurance CSV; optional Kaggle path)
  -> Preprocessing:  ColumnTransformer (scale numerics, one-hot categoricals)
                     + SMOTE on the training fold only
  -> Models:         Logistic Regression, Decision Tree, XGBoost,
                     and probability-calibrated XGBoost (selected)
  -> Explainability: SHAP (global + local), LIME, Logistic-Regression coefficients
  -> Fairness audit: Fairlearn per-group metrics + bootstrap CIs (evaluate.py)
  -> Monitoring:     Evidently data-summary and drift reports (monitoring.py)
  -> Serving:        this Gradio app, loading the saved model + preprocessor
```

## Tech stack (what is actually used)
| Concern | Library |
|---------|---------|
| Modelling | scikit-learn, XGBoost |
| Class imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP, LIME |
| Fairness | Fairlearn |
| Monitoring | Evidently |
| App | Gradio |

## Governance (NIST AI RMF)
The notebook (Section 7) scores the project against the NIST AI RMF core
functions. Current self-assessment:

| Function | Status |
|----------|--------|
| MAP | Context, target, and sensitive attributes documented |
| MEASURE | Performance and per-group fairness measured with confidence intervals |
| MANAGE | Calibration applied; sex mitigation evaluated (no out-of-sample gain); region gap documented, not mitigated |
| GOVERN | Model card published; no review cadence or named owner yet |

## Out of scope (what a production deployment would still need)
Listed for completeness -- none of this is built here: a serving API and
containerisation, authentication and access control, a model registry and
versioning, automated retraining and rollback, drift alerting, and audit logging
of predictions.
"""


def build_demo(model, preprocessor):
    """Build the Gradio interface bound to a loaded model and preprocessor."""
    predict_risk = make_predict_fn(model, preprocessor)

    with gr.Blocks(
        title="Responsible AI Insurance Underwriting",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # Insurance Risk Classification — Responsible AI Demo

        A calibrated XGBoost model that flags insurance applicants as "Good Risk"
        or "Bad Risk", served alongside its fairness audit. The tabs cover model
        performance, per-feature interpretation, and the system design.
        """)

        with gr.Tab("Make Prediction"):
            gr.Markdown(
                "Enter applicant details to get a risk classification, calibrated "
                "probabilities, and per-group fairness context."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    age = gr.Slider(
                        minimum=18, maximum=70, value=40, step=1,
                        label="Age (years)", info="Applicant's age",
                    )
                    bmi = gr.Slider(
                        minimum=12, maximum=55, value=25, step=0.1,
                        label="BMI (kg/m²)", info="Body Mass Index",
                    )
                    children = gr.Slider(
                        minimum=0, maximum=5, value=0, step=1,
                        label="Number of Children", info="Dependent children",
                    )
                with gr.Column(scale=1):
                    sex = gr.Radio(
                        choices=["male", "female"], value="male", label="Sex",
                    )
                    smoker = gr.Radio(
                        choices=["no", "yes"], value="no", label="Smoker",
                    )
                    region = gr.Radio(
                        choices=["northeast", "northwest", "southeast", "southwest"],
                        value="northeast", label="Region", info="US region",
                    )

            predict_btn = gr.Button("Predict Risk", variant="primary", size="lg")
            prediction_output = gr.Textbox(
                label="Prediction Result", interactive=False, lines=12,
                placeholder="Prediction will appear here...",
            )
            predict_btn.click(
                fn=predict_risk,
                inputs=[age, sex, bmi, children, smoker, region],
                outputs=prediction_output,
            )

            gr.Markdown(
                "The model is a decision-support tool, not an automated "
                "decision-maker: use it to inform an underwriter's judgment, not to "
                "replace it."
            )

        with gr.Tab("Model Information"):
            gr.Markdown(get_model_info())

        with gr.Tab("Feature Guide"):
            gr.Markdown(get_feature_guide())

        with gr.Tab("System Design"):
            gr.Markdown(get_system_design())

    return demo
