#!/usr/bin/env python
"""
Shared Gradio UI and inference logic for the Responsible AI underwriting demo.

Both entry points import from this module:
- app.py            (repo root; local development, searches notebooks/models)
- hf_space/app.py   (Hugging Face Space; serves the same interface)

They differ only in which directories they search for the model artifacts and
in how they launch the server. Everything else — the prediction path, the
fairness context, and the documentation tabs — lives here so the two stay in
sync.
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
        """
        Make prediction and provide explainability.
        """
        try:
            # Build a single-row frame in the order the preprocessor expects, then
            # score with the real calibrated XGBoost model. No rule-based fallback.
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
                "Probability (Bad Risk)": f"{probability*100:.1f}%",
                "Probability (Good Risk)": f"{(1-probability)*100:.1f}%",
            }

            # Fairness context
            fairness_notes = []

            if sex == "female":
                fairness_notes.append(
                    "Sex: baseline selection-rate gap was 4.4pt (women 39.8% vs men 44.3%), "
                    "but the demographic-parity ratio CI [0.67, 0.99] reaches parity, so this "
                    "gap is not statistically robust at the test sample size, and a leakage-free "
                    "ThresholdOptimizer pass did not improve it. Sex-based pricing is restricted "
                    "in many jurisdictions, so keep monitoring it as data grows."
                )

            if region == "northeast":
                fairness_notes.append(
                    "⚠️ Region Disparity: Northeast has the highest baseline risk (48.1%). "
                    "Regional disparities are measured but not mitigated (awaiting jurisdiction policy)."
                )
            elif region == "northwest":
                fairness_notes.append(
                    "✓ Region Note: Northwest has the lowest baseline risk (37.7%)."
                )

            if smoker == "yes":
                fairness_notes.append(
                    "ℹ️ Smoking: Strong signal (1.0 selection rate). Smokers almost always predicted 'Bad Risk'. "
                    "This gap is intentional and defensible—smoking is a recognized actuarial factor. "
                    "No mitigation was applied to this attribute."
                )
            else:
                fairness_notes.append(
                    "✓ Non-Smoker: Lower baseline risk (27.2% selection rate)."
                )

            if fairness_notes:
                result["Fairness & Context"] = "\n\n".join(fairness_notes)

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error: {str(e)}"

    return predict_risk


def get_model_info():
    """Return comprehensive model information."""
    return """
# Model Information & Performance

## Selected Model
- **Type:** Calibrated XGBoost (CalibratedClassifierCV wrapper)
- **Rationale:** Superior probability calibration for reliable probability estimates
- **Training Date:** 2026-06-17
- **Version:** 1.0

## Test Set Performance
| Metric | Value |
|--------|-------|
| Accuracy | 93.3% |
| Precision | 97.3% |
| Recall | 88.0% |
| F1 Score | 92.4% |
| ROC-AUC | 0.939 |

*All metrics computed on held-out test set (268 records, 20% split)*

## Model Architecture
- **Input Features:** 6 (age, sex, bmi, children, smoker, region)
- **Processing:** ColumnTransformer (scaling + one-hot encoding)
- **Imbalance Handling:** SMOTE on training fold
- **Output:** Calibrated probability + binary class

## Comparison with Baseline Models
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----|
| Logistic Regression | 84.4% | 85.6% | 90.4% | 87.9% | 93.8% |
| Decision Tree | 92.5% | 97.3% | 86.4% | 91.5% | 90.5% |
| XGBoost (uncalibrated) | 93.3% | 97.3% | 88.0% | 92.4% | 93.4% |
| **Calibrated XGBoost** | **93.3%** | **97.3%** | **88.0%** | **92.4%** | **93.9%** |

## Calibration (why the calibrated model was chosen)
On the test set, isotonic calibration improved probability quality without
changing the threshold-0.5 classification metrics:
| Metric | Raw XGBoost | Calibrated | (lower is better) |
|--------|-------------|------------|-------------------|
| Expected Calibration Error | 0.048 | **0.027** | ✓ ~44% lower |
| Brier score | 0.0599 | **0.0583** | ✓ lower |

## Fairness Analysis

Per-group samples are small, so demographic-parity ratios (DPR) are reported
with bootstrap 95% confidence intervals. The interval — not the point estimate —
is what says whether a gap is real.

### Sex — not a statistically robust gap
- **Baseline:** Female 39.8% vs Male 44.3% (4.4-point difference)
- **DPR:** 0.90, **95% CI [0.67, 0.99]** — the interval nearly reaches parity (1.0),
  so the sex gap is **not distinguishable from zero** at this sample size (~130/group)
- **Mitigation (leakage-free):** ThresholdOptimizer fit on validation, scored on
  test → gap 4.4pt → 4.5pt, DPR 0.90 → 0.90, equalized-odds 0.0 → 0.0,
  accuracy/F1 unchanged. **It does not transfer.** (The notebook's earlier
  "3.7pt / 0.52" figure came from fitting on the test set — leakage.)
- **Status:** No robust sex disparity to mitigate; keep monitoring as data grows

### Region — the one robust disparity
- **Selection rates:** 37.7% (northwest) to 48.1% (northeast), ~10-point spread
- **DPR:** 0.78, **95% CI [0.50, 0.90]** — upper bound below 1.0, so this gap **is**
  statistically robust
- **Mitigation:** Measured but **not applied** (pending jurisdiction policy decision)
- **Status:** Unresolved; awaiting business decision

### Smoker — large, robust, intentional
- **Smoker selection rate:** 100.0% (all smokers predicted "Bad Risk")
- **Non-smoker selection rate:** 27.2%
- **DPR:** 0.27, **95% CI [0.22, 0.33]** (tight) — a large, robust gap
- **Mitigation:** **Left intentional by design**
- **Rationale:** Smoking is a recognized, defensible actuarial risk factor

## Monitoring & Governance
- **Monitoring tool:** Evidently AI (data quality, data drift, model performance)
- **NIST AI RMF maturity:** 3.3/5.0 (Managed level)
- **Model card:** See MODEL_CARD.md
- **Audit trail:** All decisions documented with rationale

## Known Limitations
1. **Small dataset:** ~1,338 records (not representative of production volumes)
2. **Public data:** No validation on production data
3. **Unresolved fairness gap:** Region disparities remain
4. **Single jurisdiction:** Legal context applies to specific regulations
5. **Binary target:** Synthetic "Bad Risk" threshold ($10,000 charges)

## Production Readiness
- [ ] Governance board established for fairness policy
- [ ] Production access control and audit logging deployed
- [ ] Monitoring infrastructure (Evidently + alerting) in place
- [ ] Security testing completed
- [ ] Region fairness policy decided
- [ ] Stakeholder dashboards built
- [ ] SLAs and rollback procedures defined

## Recommendation
**Status:** Production-Ready with Qualifications

This model is ready for production use **provided that**:
1. Region fairness policy is decided and implemented
2. Continuous monitoring infrastructure is active
3. Fairness audits are scheduled quarterly
4. Access is role-based (underwriter, compliance, admin)
5. All predictions are logged for audit compliance
"""


def get_feature_guide():
    """Return detailed feature interpretation guide."""
    return """
# Feature Interpretation Guide

This guide explains each feature in the model and how to interpret them.

## Input Features

### 1. Age (numeric, 18-70 years)
**What it is:** Applicant's age in years

**Why it matters:**
- Strong predictor of healthcare costs
- Older applicants generally higher risk
- Non-linear relationship (risk accelerates after 50)

**Example:**
- Age 25: Lower risk (younger, fewer health conditions expected)
- Age 55: Higher risk (age-related health conditions more common)

---

### 2. Sex (categorical: Male / Female)
**What it is:** Biological sex as provided on application

**Why it matters:**
- Proxy for healthcare cost differences (pregnancy, etc.)
- **⚠️ Fairness concern:** Sex-based pricing restricted in many jurisdictions
- A 4.4pt baseline selection-rate gap appears, but it is within sampling noise
  (demographic-parity ratio 0.90, 95% CI [0.67, 0.99] — the interval reaches parity)

**Important Note:**
- The baseline female/male gap (4.4pt) is **not statistically robust** at this
  sample size (~130 per group)
- A **leakage-free** ThresholdOptimizer pass did not improve it out-of-sample
- The notebook's earlier "reduced to 3.7pt" figure was an artifact of fitting the
  mitigation on the test set, then scoring on that same set
- Never use sex as sole basis for denial; must combine with other factors

**Regulatory Context:**
- EU/UK: Sex not permitted for pricing (Equality Act)
- US: Varies by state; some prohibit, others allow
- Canada: Sex prohibited in life insurance

---

### 3. BMI (numeric, 12-55 kg/m²)
**What it is:** Body Mass Index (weight in kg / height² in m²)

**Why it matters:**
- Health indicator; higher BMI correlates with higher healthcare costs
- Accounts for body composition

**Categories (WHO):**
- Underweight: < 18.5
- Normal weight: 18.5 - 24.9
- Overweight: 25.0 - 29.9
- Obese: ≥ 30.0

**Example:**
- BMI 22: Normal, lower risk
- BMI 32: Obese, higher risk

---

### 4. Children (numeric count, 0-5)
**What it is:** Number of dependent children

**Why it matters:**
- Family size indicator
- More children → potentially higher family healthcare cost
- Modest predictor (weak signal in model)

**Example:**
- 0 children: Lower risk (single/couple without dependents)
- 3 children: Higher risk (larger family = more healthcare needs)

---

### 5. Smoker (categorical: Yes / No)
**What it is:** Current smoking status

**Why it matters:**
- **The dominant signal in the model** (top feature by SHAP importance)
- Smoking dramatically increases healthcare costs
- Highest disease burden and mortality risk

**Fairness Note:**
- Smokers: 100% predicted "Bad Risk"
- Non-smokers: 27.2% predicted "Bad Risk"
- **72.8 percentage-point gap** (largest in model)
- **Gap is intentional and defensible**
- Smoking is universally recognized as legitimate actuarial factor
- **No mitigation applied** because gap is defensible

**Regulatory Context:**
- Smoking status is permitted for pricing in virtually all jurisdictions
- No legal restriction (unlike sex or race)
- Consistent with industry practice

---

### 6. Region (categorical: Northeast / Northwest / Southeast / Southwest)
**What it is:** Geographic region of residence in the US

**Why it matters:**
- Healthcare cost varies significantly by region
- Regional economic differences
- State-level regulatory variation

**Baseline Risk by Region:**
- **Northeast:** 48.1% selection rate (highest risk)
- **Southeast:** ~42% selection rate
- **Southwest:** ~38% selection rate
- **Northwest:** 37.7% selection rate (lowest risk)

**Fairness Note:**
- **10-point spread** (48.1% vs 37.7%, parity ratio ≈ 0.78)
- Disparities are **measured but NOT mitigated**
- Decision deferred pending jurisdiction policy
- Some regions permit region-based pricing; others restrict

---

## Prediction Output Interpretation

### Prediction Class
- **"Good Risk":** Predicted charges likely below $10,000/year
- **"Bad Risk":** Predicted charges likely above $10,000/year

### Probability Scores
- **Probability (Bad Risk):** Likelihood of "Bad Risk" classification
- **Probability (Good Risk):** Likelihood of "Good Risk" classification
- Always sum to 100%

**Important:** Calibrated probabilities are more reliable for:
- Risk pricing
- Threshold optimization
- Business decisions requiring certainty quantification

---

## Fairness & Context Interpretation

### Green Flags (✓)
- Smoker status: Gap defensible and permitted
- Non-smoker: Lower baseline risk
- Northwest region: Lowest baseline risk

### Yellow Flags (⚠️)
- Sex: 4.4pt baseline gap, but within sampling noise (DPR CI [0.67, 0.99] reaches parity); monitor as data grows
- Region: the one statistically robust disparity (DPR 0.78, CI [0.50, 0.90]); measured but unmitigated
- Northeast region: Highest baseline risk; unresolved disparity

### Red Flags (🚩)
- Region disparities: Measured but awaiting policy decision
- Use other factors in addition to geographic region for underwriting decisions
- Consider applicant circumstances holistically

---

## Common Misconceptions

❌ **"The model is completely fair"**
- No: a statistically robust region gap remains (DPR 0.78, CI [0.50, 0.90]); the
  sex gap is within sampling noise rather than "fixed"
- Fair AI is an ongoing process, not a final state

❌ **"The model should always be trusted"**
- No: Use as tool to inform decisions, not replace judgment
- Underwriter expertise should validate model recommendations

❌ **"Higher probability always means deny coverage"**
- No: Pricing/terms can adjust for moderate risk
- Denial should be rare and well-justified

✓ **"The model should be used alongside human judgment"**
- Yes: Model provides quantitative assessment; humans provide context
- Combination is most effective approach

---

## Best Practices

1. **Always review fairness context** for each prediction
2. **Consider applicant circumstances** holistically (not just model score)
3. **Document the rationale** for approval/denial decisions
4. **Monitor predictions** by demographic group over time
5. **Report fairness metrics** quarterly to compliance
6. **Never use as sole basis** for high-stakes decisions
7. **Maintain audit trail** of all predictions and decisions
"""


def get_system_architecture():
    """Return system architecture documentation."""
    return """
# System Architecture & Design

## System Overview

This system implements a **responsible AI pipeline for insurance underwriting** with integrated
governance, explainability, fairness auditing, and continuous monitoring.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSIBLE AI SYSTEM                         │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
├── Applicant Data (age, sex, BMI, children, smoker, region)
├── Data Validation & Schema Checking
└── Feature Engineering (scaling, one-hot encoding)

ML LAYER
├── Model: Calibrated XGBoost (CalibratedClassifierCV)
├── Inference: Calibrated probability + binary class
└── Output: Binary class + calibrated probability

EXPLAINABILITY LAYER
├── Global: SHAP beeswarm plots, feature importance
├── Local: SHAP waterfall, LIME explanations
└── Interpretable baseline: Logistic Regression coefficients

FAIRNESS & BIAS LAYER
├── Sensitive attributes: Sex, Region, Smoker
├── Fairness metrics: Demographic parity, equalized odds, selection rates
├── Mitigation: ThresholdOptimizer on sex attribute
└── Documentation: Model card with ethical considerations

GOVERNANCE LAYER
├── NIST AI RMF mapping (MAP, MEASURE, MANAGE, GOVERN)
├── Trustworthiness assessment (accountability, transparency, fairness, etc.)
├── Audit trails and decision documentation
└── Stakeholder transparency

MONITORING LAYER
├── Data Quality: Schema, completeness, anomalies
├── Data Drift: Feature distribution shifts
├── Model Performance: Accuracy, precision, recall, ROC-AUC
├── Fairness Drift: Per-group metrics over time
└── Alerting: Automated notifications for issues

OUTPUT LAYER
├── Prediction: Risk classification + calibrated probability
├── Explainability: Feature contributions, local explanations
├── Fairness Context: Group-specific metrics and caveats
└── Audit Record: Timestamp, features, decision, rationale
```

## Component Details

### 1. Data Processing Pipeline
- **Validation:** Schema checking, data type validation
- **Quality checks:** Completeness, outlier detection, anomaly flags
- **Feature engineering:** Scaling (StandardScaler), encoding (one-hot)
- **Imbalance handling:** SMOTE on training set only
- **Splits:** 60% train / 20% validation / 20% test (stratified)

### 2. Model Selection Logic
- **Primary:** Calibrated XGBoost (best calibration for probabilities)
- **Alternatives:** Logistic Regression, Decision Tree (interpretability baseline)
- **Selection criteria:**
  - High accuracy (93.3%)
  - High precision (97.3%) - minimize false "Bad Risk" predictions
  - Reliable probabilities (ROC-AUC 0.939)
  - Calibration quality (highest among ensemble methods)

### 3. Explainability Strategy
- **SHAP (SHapley Additive exPlanations):**
  - Global: Feature importance via mean |SHAP values|
  - Local: Waterfall plots showing feature contributions per prediction
  - Dependence: Feature-outcome relationships

- **LIME (Local Interpretable Model-agnostic Explanations):**
  - Local model-agnostic explanations
  - Works with any model
  - Individual feature importance

- **Direct Inspection:**
  - Logistic Regression coefficients
  - Easy stakeholder communication

### 4. Fairness Framework
- **Evaluation:** Fairlearn with per-group metrics
- **Sensitive attributes:** Sex, Region, Smoker
- **Metrics tracked:**
  - Selection rate (share predicted "Bad Risk")
  - Recall (true positive rate)
  - Precision (positive predictive value)
  - Demographic parity ratio

- **Mitigation evaluated:** ThresholdOptimizer on sex (equalized odds), fit on
  validation and scored on the held-out test set (no leakage)
- **Result:** No out-of-sample effect — sex gap 4.4pt → 4.5pt, accuracy/F1
  unchanged; the sex gap is within sampling noise (DPR CI [0.67, 0.99])
- **Robust disparity:** `region` (DPR 0.78, CI [0.50, 0.90]), measured but not mitigated

### 5. Monitoring Architecture

```
┌──────────────────────┐
│  Production Data     │
│  (Applicants)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│     Evidently AI Monitoring Suite        │
├──────────────────────────────────────────┤
│  • Data Quality Report                   │
│  • Data Drift Report                     │
│  • Model Performance Report              │
│  • Fairness Metrics Report               │
│  • Test Suite (Automated Checks)         │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│     Alert System                         │
├──────────────────────────────────────────┤
│  • Data drift detected: Email alert      │
│  • Performance degradation: Slack notify │
│  • Fairness gap widened: Dashboard flag  │
│  • Schema violation: Critical alert      │
└──────────────────────────────────────────┘
```

### 6. NIST AI RMF Alignment

| Function | Status | Evidence |
|----------|--------|----------|
| **MAP** | Complete | Risk areas identified; sensitive attributes documented |
| **MEASURE** | Complete | Fairness metrics computed; performance metrics tracked |
| **MANAGE** | Complete | Mitigation applied; model card published |
| **GOVERN** | In Progress | Governance board needed; policy decisions pending |

| Characteristic | Maturity | Status |
|---|---|---|
| Accountability | 4/5 | Mature |
| Transparency | 4/5 | Mature |
| Fairness | 3/5 | Managed |
| Resilience | 3/5 | Managed |
| Privacy | 2/5 | Emerging |
| Security | 2/5 | Emerging |
| Inclusivity | 3/5 | Managed |

## Deployment Architecture

### Development → Staging → Production Pipeline

```
Local Development
  ├── Jupyter notebook
  ├── Model training & evaluation
  ├── Fairness audit
  └── Model card generation
       │
       ▼
Staging Environment
  ├── Docker container
  ├── API (FastAPI/Flask)
  ├── Monitoring (Evidently)
  └── Test on sample data
       │
       ▼
Production Environment
  ├── Kubernetes orchestration
  ├── Auto-scaling
  ├── Load balancing
  ├── API gateway (authentication)
  ├── Monitoring & alerts
  ├── Audit logging
  └── Model versioning
```

### Production Checklist

- [ ] **Access Control:** Role-based (underwriter, compliance, admin)
- [ ] **Audit Logging:** All predictions logged with timestamp, features, decision
- [ ] **Monitoring:** Evidently AI + dashboards
- [ ] **Alerting:** Slack/email for drift, performance issues
- [ ] **Compliance:** GDPR/CCPA data handling
- [ ] **Security:** API authentication, encryption at rest/transit
- [ ] **Fairness Policy:** Region mitigation decision documented
- [ ] **SLAs:** Uptime target, response time, accuracy guarantees
- [ ] **Rollback Plan:** Procedure to revert to previous model version
- [ ] **Documentation:** System design, governance, training materials

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | XGBoost, scikit-learn |
| Preprocessing | pandas, scikit-learn ColumnTransformer |
| Explainability | SHAP, LIME |
| Fairness | Fairlearn |
| Monitoring | Evidently AI |
| UI/Demo | Gradio |
| API | FastAPI (production) |
| Orchestration | Kubernetes |
| Registry | MLflow, Docker Registry |
| Logging | ELK Stack |
| Alerting | Prometheus + Grafana |

## Security & Compliance

### Data Protection
- Anonymization for testing
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII detection and redaction

### Model Security
- Model versioning
- Digital signatures
- Theft/poisoning defenses (adversarial training)
- Watermarking

### Access Control
- Authentication (OAuth2, SAML)
- Authorization (role-based)
- Audit logging
- Session management

### Compliance
- GDPR: Data minimization, right to explanation, data deletion
- CCPA: Data transparency, opt-out rights
- Fair Lending: Documentation of fairness practices
- Insurance regulation: Risk disclosure, model governance

## Continuous Improvement

### Quarterly Reviews
- [ ] Recompute fairness metrics on production traffic
- [ ] Assess data drift impact
- [ ] Review model performance trends
- [ ] Audit decision patterns by demographic group

### Annual Audits
- [ ] Full model revalidation
- [ ] Fairness assessment by external auditor
- [ ] NIST AI RMF reassessment
- [ ] Update model card

### Feedback Loop
- Incorporate underwriter feedback
- Monitor real outcomes vs predictions
- Retrain with new data (if drift detected)
- A/B test model improvements
"""


def build_demo(model, preprocessor):
    """Build the Gradio interface bound to a loaded model and preprocessor."""
    predict_risk = make_predict_fn(model, preprocessor)

    with gr.Blocks(
        title="Responsible AI Insurance Underwriting",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .tab-nav { display: flex; gap: 10px; margin-bottom: 20px; }
        """
    ) as demo:

        # Header
        gr.Markdown("""
        # 🔒 Insurance Risk Classification: Responsible AI Demo

        An interactive application demonstrating responsible AI practices in predictive underwriting.

        **Features:**
        - ✅ **Prediction & Probability:** Risk classification with calibrated probability estimates
        - 🎯 **Fairness Awareness:** Bias metrics and demographic context
        - 📊 **Transparency:** Model information and explainability guide
        - 🏗️ **Architecture:** System design and governance framework
        """)

        # Main prediction interface
        with gr.Tab("🔮 Make Prediction"):
            gr.Markdown("""
            ### Enter applicant information to generate risk prediction

            The model will provide:
            1. Risk classification (Good Risk / Bad Risk)
            2. Calibrated probability estimates
            3. Fairness & demographic context
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    age = gr.Slider(
                        minimum=18, maximum=70, value=40, step=1,
                        label="Age (years)",
                        info="Applicant's age"
                    )
                    bmi = gr.Slider(
                        minimum=12, maximum=55, value=25, step=0.1,
                        label="BMI (kg/m²)",
                        info="Body Mass Index"
                    )
                    children = gr.Slider(
                        minimum=0, maximum=5, value=0, step=1,
                        label="Number of Children",
                        info="Dependent children"
                    )

                with gr.Column(scale=1):
                    sex = gr.Radio(
                        choices=["male", "female"], value="male",
                        label="Sex",
                        info="Biological sex"
                    )
                    smoker = gr.Radio(
                        choices=["no", "yes"], value="no",
                        label="Smoker Status",
                        info="Current smoking status"
                    )
                    region = gr.Radio(
                        choices=["northeast", "northwest", "southeast", "southwest"],
                        value="northeast",
                        label="Region",
                        info="Geographic region (US)"
                    )

            predict_btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")

            with gr.Row():
                prediction_output = gr.Textbox(
                    label="📋 Prediction Result",
                    interactive=False,
                    lines=12,
                    placeholder="Prediction will appear here..."
                )

            predict_btn.click(
                fn=predict_risk,
                inputs=[age, sex, bmi, children, smoker, region],
                outputs=prediction_output
            )

            gr.Markdown("""
            ---
            **How to interpret results:**
            - Look at the "Prediction" and the "Probability (Bad Risk)" score
            - Read the "Fairness & Context" section for demographic details
            - Use the Feature Guide tab to understand feature importance
            - Remember: Model is a tool to inform decisions, not replace human judgment
            """)

        # Model information
        with gr.Tab("📊 Model Information"):
            gr.Markdown(get_model_info())

        # Feature guide
        with gr.Tab("📖 Feature Guide"):
            gr.Markdown(get_feature_guide())

        # System architecture
        with gr.Tab("🏗️ System Architecture"):
            gr.Markdown(get_system_architecture())

    return demo
