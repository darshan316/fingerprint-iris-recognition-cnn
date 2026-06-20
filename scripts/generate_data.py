#!/usr/bin/env python3
"""
Generate the synthetic population and write it under data/.

Also drops a handful of ready-to-use probe images into samples/ so you can
exercise the GUI immediately (a genuine pair, and a deliberately mismatched
pair), and renders the app icon used by the GUI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402
from src.data.synth import (  # noqa: E402
    generate_population, synth_fingerprint, synth_iris, subject_id, subject_name,
)

SAMPLES = config.ROOT / "samples"
ASSETS = config.ROOT / "assets"


def _progress(done, total):
    pct = int(100 * done / total)
    bar = "#" * (pct // 4)
    print(f"\r  [{bar:<25}] {done}/{total}", end="", flush=True)


def export_probes():
    """Save unseen captures (sample index far outside the training range).

    These are rendered larger than the CNN input so they look crisp in the
    GUI preview; the matcher downsizes them on the way in.
    """
    SAMPLES.mkdir(parents=True, exist_ok=True)
    probe = 9001  # a "fresh" capture the models never trained on
    disp = 256

    def save(arr, name):
        Image.fromarray(arr).save(SAMPLES / name)

    # a genuine pair: same subject's finger + eye
    save(synth_fingerprint(2, probe, size=disp), "genuine_fingerprint_SUBJ_02.png")
    save(synth_iris(2, probe, size=disp), "genuine_iris_SUBJ_02.png")

    # a mismatched pair: one person's finger, another's eye
    save(synth_fingerprint(4, probe, size=disp), "mismatch_fingerprint_SUBJ_04.png")
    save(synth_iris(7, probe, size=disp), "mismatch_iris_SUBJ_07.png")

    # someone who was never enrolled (index beyond the population)
    stranger = config.NUM_SUBJECTS + 3
    save(synth_fingerprint(stranger, probe, size=disp), "stranger_fingerprint.png")
    save(synth_iris(stranger, probe, size=disp), "stranger_iris.png")
    print(f"  wrote demo probes to {SAMPLES}")


def render_icon():
    """A small two-tone icon: a fingerprint swirl beside an iris."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    teal, ink = (45, 212, 191, 255), (15, 23, 42, 255)
    d.rounded_rectangle([8, 8, S - 8, S - 8], radius=48, fill=ink)

    # fingerprint: concentric arcs on the left
    cx, cy = 92, 132
    for i, r in enumerate(range(18, 78, 12)):
        d.arc([cx - r, cy - r, cx + r, cy + r], start=20, end=320,
              fill=teal, width=6)
    # iris: ring + pupil on the right
    ex, ey, er = 176, 132, 44
    d.ellipse([ex - er, ey - er, ex + er, ey + er], outline=teal, width=8)
    d.ellipse([ex - 16, ey - 16, ex + 16, ey + 16], fill=teal)

    img.save(ASSETS / "app_icon.png")
    print(f"  wrote icon to {ASSETS / 'app_icon.png'}")


def main():
    n, k = config.NUM_SUBJECTS, config.SAMPLES_PER_SUBJECT
    print(f"Generating {n} subjects x {k} captures per modality "
          f"({n * k * 2} images, {config.IMG_SIZE}px)...")
    generate_population(n, k, config.FINGER_DIR, config.IRIS_DIR,
                        size=config.IMG_SIZE, progress=_progress)
    print()
    export_probes()
    render_icon()
    print("\nEnrolled population:")
    for i in range(n):
        print(f"  {subject_id(i)}  {subject_name(i)}")
    print("\nData generation complete.")


if __name__ == "__main__":
    main()
