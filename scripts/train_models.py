#!/usr/bin/env python3
"""Train both CNNs on the generated data and save them under models/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.train import train_all  # noqa: E402


def main():
    if not any(config.FINGER_DIR.iterdir()):
        sys.exit("No data found. Run scripts/generate_data.py first.")

    results = train_all(verbose=2)

    print("\n" + "=" * 46)
    print(f"{'modality':<14}{'train':>7}{'val':>6}{'val acc':>10}")
    print("-" * 46)
    for r in results:
        print(f"{r['modality']:<14}{r['train']:>7}{r['val']:>6}"
              f"{r['val_accuracy']*100:>9.1f}%")
    print("=" * 46)
    print(f"Models saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
