"""
The matcher: given an uploaded fingerprint and iris, decide whether they
belong to the same enrolled person.

Pipeline per modality:
  image -> preprocess -> CNN embedding -> cosine similarity to every stored
  template -> best-matching subject + score.

Then we fuse. Score-level fusion (a weighted sum of the two similarity
vectors) is the standard, robust choice for multimodal biometrics: it does
not need the two networks to share a feature space, just comparable scores.
The final call has three outcomes the GUI cares about:

  match         both modalities agree on a subject and the fused score is
                high enough -> same person, access granted
  different      both modalities are confident but name different subjects
                -> the print and the eye come from two different people
  unrecognized   at least one modality can't confidently place its input in
                the gallery -> not enrolled, or a poor capture
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tensorflow import keras

import config
from src.data.dataset import preprocess
from src.models.cnn import embedding_model


@dataclass
class ModalityResult:
    modality: str
    best_id: str
    best_name: str
    best_score: float
    recognized: bool
    ranking: list                      # [(subject_id, name, score), ...] top-k
    scores: np.ndarray = field(repr=False, default=None)


@dataclass
class VerificationResult:
    decision: str                      # "match" | "different" | "unrecognized"
    headline: str
    detail: str
    confidence: float                  # 0..1
    fingerprint: ModalityResult
    iris: ModalityResult
    fused_id: str
    fused_name: str
    fused_score: float


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class BiometricMatcher:
    """Loads the trained models + enrollment gallery and answers verify()."""

    def __init__(self):
        db = np.load(config.ENROLL_DB, allow_pickle=True)
        fp_labels = list(db["fingerprint_labels"])
        ir_labels = list(db["iris_labels"])
        if fp_labels != ir_labels:
            raise ValueError("fingerprint and iris galleries are not aligned; "
                             "re-run enrollment.")
        self.ids = fp_labels
        self.names = list(db["names"])
        self.templates = {
            "fingerprint": db["fingerprint_templates"],
            "iris": db["iris_templates"],
        }

        self._embed = {}
        for modality in ("fingerprint", "iris"):
            model = keras.models.load_model(str(config.MODEL_DIR / f"{modality}.keras"))
            self._embed[modality] = embedding_model(model)

    # -- single modality ---------------------------------------------------
    def match_modality(self, modality: str, image, top_k: int = 3) -> ModalityResult:
        x = preprocess(image)[np.newaxis, ...]
        q = _l2(self._embed[modality].predict(x, verbose=0))[0]
        sims = np.clip(self.templates[modality] @ q, 0.0, 1.0)

        order = np.argsort(sims)[::-1]
        ranking = [(self.ids[i], self.names[i], float(sims[i])) for i in order[:top_k]]
        best = int(order[0])
        return ModalityResult(
            modality=modality,
            best_id=self.ids[best],
            best_name=self.names[best],
            best_score=float(sims[best]),
            recognized=bool(sims[best] >= config.PER_MODALITY_THRESHOLD),
            ranking=ranking,
            scores=sims,
        )

    # -- the actual question -----------------------------------------------
    def verify(self, fingerprint_image, iris_image) -> VerificationResult:
        fp = self.match_modality("fingerprint", fingerprint_image)
        ir = self.match_modality("iris", iris_image)

        wf, wi = config.FUSION_WEIGHTS["fingerprint"], config.FUSION_WEIGHTS["iris"]
        fused = wf * fp.scores + wi * ir.scores
        fb = int(np.argmax(fused))
        fused_id, fused_name, fused_score = self.ids[fb], self.names[fb], float(fused[fb])

        both_ok = fp.recognized and ir.recognized
        agree = fp.best_id == ir.best_id

        if both_ok and agree and fused_score >= config.FUSION_THRESHOLD:
            decision = "match"
            headline = f"Same person — {fp.best_name}"
            detail = (f"Fingerprint and iris both identify {fp.best_name} "
                      f"({fp.best_id}). Fused confidence {fused_score*100:.1f}%.")
            confidence = fused_score
        elif both_ok and not agree:
            decision = "different"
            headline = "Different people"
            detail = (f"Fingerprint matches {fp.best_name} ({fp.best_id}) at "
                      f"{fp.best_score*100:.1f}%, but iris matches {ir.best_name} "
                      f"({ir.best_id}) at {ir.best_score*100:.1f}%. The two "
                      f"samples do not come from the same enrolled subject.")
            confidence = max(fp.best_score, ir.best_score)
        else:
            decision = "unrecognized"
            weak = "fingerprint" if fp.best_score <= ir.best_score else "iris"
            headline = "Not recognised"
            detail = (f"Could not confidently match the {weak} to anyone in the "
                      f"gallery (best {min(fp.best_score, ir.best_score)*100:.1f}% "
                      f"vs threshold {config.PER_MODALITY_THRESHOLD*100:.0f}%). "
                      f"The subject may not be enrolled, or the capture is poor.")
            confidence = min(fp.best_score, ir.best_score)

        return VerificationResult(
            decision=decision,
            headline=headline,
            detail=detail,
            confidence=float(confidence),
            fingerprint=fp,
            iris=ir,
            fused_id=fused_id,
            fused_name=fused_name,
            fused_score=fused_score,
        )
