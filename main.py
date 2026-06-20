#!/usr/bin/env python3
"""
Entry point for the BioVerify desktop app.

    python main.py

If the models haven't been trained yet the window still opens, but it will
tell you to run scripts/run_all.py first. That script generates the synthetic
data, trains both CNNs and builds the enrollment gallery.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    try:
        from gui.app import launch
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", str(exc))
        sys.exit(
            f"Missing dependency: {missing}\n"
            "Install requirements first:  pip install -r requirements.txt"
        )
    launch()


if __name__ == "__main__":
    main()
