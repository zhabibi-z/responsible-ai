# reports/

Evaluation and monitoring artifacts.

**Committed (documentation asset):**
- `calibration_reliability.png` — raw vs. calibrated XGBoost reliability diagram, rendered in the top-level [README](../README.md).

**Generated and git-ignored** — regenerate locally:

```bash
python src/underwriting/evaluate.py     # fairness CIs + calibration metrics + reliability diagram + evaluation_results.json
python src/underwriting/monitoring.py   # Evidently data-summary and drift reports (HTML)
```

The Evidently HTML reports are ~3–4 MB each and do not render on GitHub, so they
are not version-controlled — run the scripts above to view them in a browser.
