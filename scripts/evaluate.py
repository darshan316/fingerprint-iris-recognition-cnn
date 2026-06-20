#!/usr/bin/env python3
"""
Evaluate the end-to-end system on fresh, unseen captures.

We generate probe captures the networks never trained on (sample indices far
outside the training range) and run them through verify() in three regimes:

  genuine    same subject's fingerprint + iris   -> expect "match"
  cross      subject A's fingerprint + B's iris   -> expect "different"
  stranger   a subject who was never enrolled     -> expect "unrecognized"

It prints accuracy per regime plus the genuine/impostor score separation,
which is what you'd use to set the thresholds in config.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data.synth import synth_fingerprint, synth_iris  # noqa: E402
from src.matcher import BiometricMatcher  # noqa: E402

PROBE_BASE = 9100          # capture ids reserved for evaluation
N_PROBES = 4               # probes per subject


def main():
    rng = np.random.default_rng(7)
    m = BiometricMatcher()
    n = config.NUM_SUBJECTS

    genuine_ok, genuine_scores = 0, []
    cross_ok, cross_total = 0, 0
    stranger_ok, stranger_total = 0, 0

    for s in range(n):
        for p in range(N_PROBES):
            cap = PROBE_BASE + p
            fp = synth_fingerprint(s, cap)
            ir = synth_iris(s, cap)

            # genuine
            r = m.verify(fp, ir)
            genuine_scores.append(r.fused_score)
            genuine_ok += (r.decision == "match" and r.fingerprint.best_id ==
                           f"{config.SUBJECT_PREFIX}_{s:02d}")

            # cross: keep this finger, borrow a different subject's iris
            other = (s + 1 + int(rng.integers(n - 1))) % n
            r2 = m.verify(fp, synth_iris(other, cap + 1))
            cross_total += 1
            cross_ok += (r2.decision == "different")

    # strangers (never enrolled)
    for j in range(n):
        sj = config.NUM_SUBJECTS + 1 + j
        r = m.verify(synth_fingerprint(sj, PROBE_BASE), synth_iris(sj, PROBE_BASE))
        stranger_total += 1
        stranger_ok += (r.decision == "unrecognized")

    g_total = n * N_PROBES
    genuine_scores = np.array(genuine_scores)

    print("\n================ evaluation ================")
    print(f"genuine pairs   : {genuine_ok}/{g_total} accepted "
          f"({100*genuine_ok/g_total:.1f}%)")
    print(f"cross pairs     : {cross_ok}/{cross_total} flagged different "
          f"({100*cross_ok/cross_total:.1f}%)")
    print(f"stranger pairs  : {stranger_ok}/{stranger_total} rejected "
          f"({100*stranger_ok/stranger_total:.1f}%)")
    print("--------------------------------------------")
    print(f"genuine fused score: mean {genuine_scores.mean():.3f} "
          f"min {genuine_scores.min():.3f}")
    print(f"thresholds in use  : per-modality {config.PER_MODALITY_THRESHOLD}, "
          f"fusion {config.FUSION_THRESHOLD}")
    print("============================================")


if __name__ == "__main__":
    main()
