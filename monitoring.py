#!/usr/bin/env python
"""Evidently monitoring for the Responsible AI underwriting project.

Reads the same insurance dataset the notebook uses, reproduces the train/test
feature split, and writes Evidently data-summary and drift reports to reports/.
Reference is the training feature set; current is the held-out test set. A
perturbed copy of the test set provides the drift-alarm scenario. No synthetic
data is generated.

Usage:
    python monitoring.py

Outputs:
    HTML reports in the reports/ directory.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Matches the notebook's Config.
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)
RISK_THRESHOLD = 10000.0
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
NUMERICAL = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
FEATURES = NUMERICAL + CATEGORICAL
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
# The notebook runs from notebooks/ and saves artifacts under notebooks/models.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "models")


def load_feature_frames():
    """Reproduce the notebook's train and test feature frames from the CSV."""
    df = pd.read_csv(DATA_URL)
    df["risk_level"] = (df["charges"] > RISK_THRESHOLD).astype(int)
    X = df[FEATURES]
    y = df["risk_level"]

    # Same two-stage split as Section 3 of the notebook (train 60% / test 20%).
    X_train_val, X_test, y_train_val, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, _ = train_test_split(
        X_train_val, test_size=VAL_SIZE, random_state=RANDOM_SEED,
        stratify=y_train_val,
    )
    return X_train, X_test


def drifted_count(result):
    """Return (count, share) of drifted columns from a DataDriftPreset result."""
    for metric in result.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            value = metric["value"]
            return int(value["count"]), float(value["share"])
    return None, None


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Loading the saved preprocessor is optional here (drift runs on raw
    # features), but confirm it exists so the script and notebook stay in sync.
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if os.path.exists(preproc_path):
        print(f"Found saved preprocessor: {preproc_path}")
    else:
        print("Saved preprocessor not found; run the notebook to create it. "
              "Drift reports below still use the real CSV features.")

    X_train, X_test = load_feature_frames()
    print(f"Reference (train) shape: {X_train.shape}; current (test) shape: {X_test.shape}")

    data_definition = DataDefinition(
        numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL
    )
    ref = Dataset.from_pandas(X_train, data_definition=data_definition)
    cur = Dataset.from_pandas(X_test, data_definition=data_definition)

    summary = Report(metrics=[DataSummaryPreset()]).run(ref, cur)
    summary_path = os.path.join(REPORTS_DIR, "monitoring_data_summary.html")
    summary.save_html(summary_path)
    print(f"Saved {summary_path}")

    drift = Report(metrics=[DataDriftPreset()]).run(ref, cur)
    drift_path = os.path.join(REPORTS_DIR, "monitoring_data_drift.html")
    drift.save_html(drift_path)
    n_drift, share_drift = drifted_count(drift)
    print(f"Saved {drift_path}  (drifted {n_drift}/{len(FEATURES)}, share {share_drift:.2f})")

    # Alarm scenario: ageing, higher-BMI book of business (deterministic shift).
    drifted = X_test.copy()
    drifted["age"] = drifted["age"] + 8
    drifted["bmi"] = drifted["bmi"] * 1.15
    alarm_cur = Dataset.from_pandas(drifted, data_definition=data_definition)
    alarm = Report(metrics=[DataDriftPreset()]).run(ref, alarm_cur)
    alarm_path = os.path.join(REPORTS_DIR, "monitoring_data_drift_alarm.html")
    alarm.save_html(alarm_path)
    n_alarm, share_alarm = drifted_count(alarm)
    print(f"Saved {alarm_path}  (drifted {n_alarm}/{len(FEATURES)}, share {share_alarm:.2f})")


if __name__ == "__main__":
    main()
