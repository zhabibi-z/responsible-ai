#!/usr/bin/env python
"""
Hugging Face Space entry point for the Responsible AI underwriting demo.

The UI and inference logic live in the repo-root underwriting_demo.py module,
which is shared with the local app.py. This script only resolves the model path
(the Space ships the artifacts under hf_space/models/) and launches the server.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Shared module lives at the repo root, one level up from hf_space/.
sys.path.insert(0, os.path.dirname(_HERE))

from underwriting_demo import load_model_artifacts, build_demo  # noqa: E402

# Prefer the artifacts bundled with the Space; fall back to the notebook output
# when running from a full repo checkout.
_MODEL_DIRS = [
    os.path.join(_HERE, "models"),
    os.path.join(_HERE, "..", "notebooks", "models"),
]

# Load the real artifacts and build the interface at module level so Hugging
# Face Spaces can launch it directly.
MODEL, PREPROCESSOR = load_model_artifacts(_MODEL_DIRS)
demo = build_demo(MODEL, PREPROCESSOR)


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Responsible AI Predictive Underwriting - Gradio Application")
    print("="*80 + "\n")

    print("Interface ready. Launching Gradio server...")
    demo.launch()


if __name__ == "__main__":
    main()
