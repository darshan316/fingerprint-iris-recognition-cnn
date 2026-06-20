#!/usr/bin/env python3
"""Build the enrollment gallery (template per subject per modality)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.enrollment import build_enrollment  # noqa: E402


def main():
    for m in ("fingerprint", "iris"):
        if not (config.MODEL_DIR / f"{m}.keras").exists():
            sys.exit("Models missing. Run scripts/train_models.py first.")

    summary = build_enrollment(progress=lambda m: print(f"  embedding {m} captures..."))
    print(f"\nEnrolled {summary['enrolled']} subjects.")
    print(f"Gallery written to {summary['path']}")


if __name__ == "__main__":
    main()
