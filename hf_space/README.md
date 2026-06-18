---
title: Responsible AI Predictive Underwriting
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Responsible AI Predictive Underwriting

Interactive demo of an insurance risk-classification model with explainability
and fairness context. It loads the calibrated XGBoost model and preprocessor
trained in the companion notebook and serves predictions through Gradio.

- **Inputs:** age, sex, BMI, children, smoker, region.
- **Output:** a "Good Risk" / "Bad Risk" label with the model's probability,
  plus fairness notes for sex, region, and smoker drawn from the project's
  fairness audit.

The model is the unmitigated calibrated XGBoost classifier. It is a learning
and portfolio demonstration, not a tool for real underwriting decisions: it is
trained on a small public dataset and has known, documented fairness gaps
(region disparity unmitigated; sex gap reduced but not closed).
