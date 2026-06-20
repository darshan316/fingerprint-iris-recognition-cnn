#!/usr/bin/env python3
"""
One command to take the project from empty to ready: generate data, train
both CNNs, build the enrollment gallery, then evaluate.

    python scripts/run_all.py

Skips data generation if data already exists (pass --fresh to force it).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _data_ok() -> bool:
    """True only if data exists *and* matches the configured image size."""
    if not (config.FINGER_DIR.exists() and any(config.FINGER_DIR.iterdir())):
        return False
    from PIL import Image
    sample = next(config.FINGER_DIR.rglob("*.png"), None)
    return sample is not None and Image.open(sample).size == (config.IMG_SIZE,
                                                              config.IMG_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="regenerate data even if it already exists")
    args = ap.parse_args()

    t0 = time.time()

    if args.fresh or not _data_ok():
        from scripts import generate_data
        generate_data.main()
    else:
        print("Data already present and matches config, skipping generation "
              "(use --fresh to redo).")

    from src.train import train_all
    print("\n--- training ---")
    train_all(verbose=2)

    from src.enrollment import build_enrollment
    print("\n--- enrollment ---")
    summary = build_enrollment(progress=lambda m: print(f"  embedding {m}..."))
    print(f"enrolled {summary['enrolled']} subjects")

    from scripts import evaluate
    evaluate.main()

    print(f"\nAll done in {time.time() - t0:.0f}s. Launch the GUI with: python main.py")


if __name__ == "__main__":
    main()
