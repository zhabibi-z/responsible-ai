"""Offline smoke tests for the underwriting project.

These run without network access or retraining: they exercise the served
prediction path against the committed artifacts and check the pure helper
functions in evaluate.py. Run with:  pytest -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
from underwriting_demo import load_model_artifacts, make_predict_fn, FEATURE_ORDER

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIRS = [os.path.join(_HERE, "..", "notebooks", "models")]


@pytest.fixture(scope="module")
def predict_fn():
    model, preprocessor = load_model_artifacts(_MODEL_DIRS)
    return make_predict_fn(model, preprocessor)


def test_feature_order():
    assert FEATURE_ORDER == ["age", "bmi", "children", "sex", "smoker", "region"]


def test_prediction_returns_valid_label(predict_fn):
    out = predict_fn(age=40, sex="male", bmi=28.0, children=1,
                     smoker="no", region="northeast")
    assert ("Good Risk" in out) or ("Bad Risk" in out)
    assert "Probability (Bad Risk)" in out


def test_smoker_scores_higher_than_nonsmoker(predict_fn):
    """A heavy smoker should not score as lower risk than an identical non-smoker."""
    import json
    base = dict(age=55, sex="male", bmi=34.0, children=0, region="southeast")
    p_smoker = float(json.loads(predict_fn(smoker="yes", **base))
                     ["Probability (Bad Risk)"].rstrip("%"))
    p_non = float(json.loads(predict_fn(smoker="no", **base))
                  ["Probability (Bad Risk)"].rstrip("%"))
    assert p_smoker >= p_non


def test_ece_perfectly_calibrated_is_zero():
    # Probabilities exactly equal to outcomes -> zero calibration error.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert evaluate.expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_pred = np.array([1] * 40 + [0] * 60)  # selection rate 0.40
    point = y_pred.mean()
    lo, hi = evaluate.bootstrap_ci(
        lambda idx: y_pred[idx].mean(), n=len(y_pred), rng=rng, n_boot=500)
    assert lo <= point <= hi
