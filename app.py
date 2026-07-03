#!/usr/bin/env python
"""
Interactive Gradio Application for Responsible AI Predictive Underwriting

Entry point for both local runs and the Hugging Face Space. The UI and
inference logic live in src/underwriting/demo.py; this script only resolves the
model path and launches the server. It is kept at the repo root because the
Hugging Face Space uses `app_file: app.py` (see the README front matter).

Usage:
    python app.py

Then open http://localhost:7860 in your browser.

Note:
    Run the Jupyter notebook first so the trained model and preprocessor are
    saved under notebooks/models/. This app loads those real artifacts; it has
    no synthetic prediction path.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Make the src/ package importable without an install step (needed on the
# Hugging Face Space, which only runs `pip install -r requirements.txt`).
sys.path.insert(0, os.path.join(_HERE, "src"))

from underwriting.demo import load_model_artifacts, build_demo  # noqa: E402

# The notebook saves the trained artifacts under notebooks/models.
_MODEL_DIRS = [
    os.path.join(_HERE, "notebooks", "models"),
    os.path.join(_HERE, "models"),
]

# Load the real artifacts and build the interface at import time.
MODEL, PREPROCESSOR = load_model_artifacts(_MODEL_DIRS)
demo = build_demo(MODEL, PREPROCESSOR)


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Responsible AI Predictive Underwriting - Gradio Application")
    print("="*80 + "\n")

    # Guard the launch so importing this module (or running headless) does not
    # block. Set RUN_APP=0 to load the model without starting the server.
    run_app = os.environ.get("RUN_APP", "1") == "1"
    if not run_app:
        print("RUN_APP=0 -> model loaded and interface built, not launching.")
        return

    print("Starting Gradio server on http://localhost:7860 (Ctrl+C to stop).")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
