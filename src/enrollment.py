"""
Enrollment: turning trained CNNs into a gallery of stored templates.

For each enrolled subject we run all of their captures through the embedding
model, L2-normalise, and average into one template vector per modality. The
average is a decent prototype of the subject and smooths out per-capture
noise. These templates are the "stored data" the matcher compares against.

The result is a single .npz holding, per modality, a (num_subjects, dim)
template matrix and the aligned subject ids, plus the display names.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras

import config
from src.data.dataset import load_modality
from src.data.synth import subject_name
from src.models.cnn import embedding_model

MODALITY_DIRS = {
    "fingerprint": config.FINGER_DIR,
    "iris": config.IRIS_DIR,
}


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _templates_for(modality: str):
    model = keras.models.load_model(str(config.MODEL_DIR / f"{modality}.keras"))
    embed = embedding_model(model)

    X, y, subjects = load_modality(MODALITY_DIRS[modality])
    emb = _l2(embed.predict(X, batch_size=config.BATCH_SIZE, verbose=0))

    templates = np.zeros((len(subjects), emb.shape[1]), dtype="float32")
    for label in range(len(subjects)):
        templates[label] = emb[y == label].mean(axis=0)
    templates = _l2(templates)               # renormalise the prototype
    return templates, subjects


def build_enrollment(progress=None) -> dict:
    store, summary = {}, {}
    subjects_ref = None

    for i, modality in enumerate(MODALITY_DIRS):
        if progress:
            progress(modality)
        templates, subjects = _templates_for(modality)
        store[f"{modality}_templates"] = templates
        store[f"{modality}_labels"] = np.array(subjects)
        summary[modality] = len(subjects)
        subjects_ref = subjects

    store["names"] = np.array([subject_name(i) for i in range(len(subjects_ref))])

    config.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.ENROLL_DB, **store)
    summary["enrolled"] = len(subjects_ref)
    summary["path"] = str(config.ENROLL_DB)
    return summary
