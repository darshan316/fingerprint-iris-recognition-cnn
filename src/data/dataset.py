"""
Turning images on disk into arrays the network can eat, and the single-image
preprocessing the GUI reuses at inference time.

Keeping preprocessing in one place matters: whatever we do to training images
(grayscale, resize, scale to [0, 1]) has to happen identically to an uploaded
image, or the embeddings won't live in the same space and matching falls apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import IMG_SIZE


def preprocess(img, size: int = IMG_SIZE) -> np.ndarray:
    """Normalise a single image to a (size, size, 1) float32 tensor in [0, 1].

    Accepts a file path, a PIL image, or a raw numpy array so the same call
    works for the dataset, the enrollment step and the GUI upload.
    """
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[..., np.newaxis]


def load_modality(modality_dir) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load every capture under a modality directory.

    Expects <modality_dir>/<subject_id>/<something>.png. Returns the image
    tensor X, integer labels y, and the ordered list of subject ids so a
    label index can be mapped back to "SUBJ_05".
    """
    modality_dir = Path(modality_dir)
    subjects = sorted(p.name for p in modality_dir.iterdir() if p.is_dir())
    if not subjects:
        raise FileNotFoundError(
            f"No subject folders in {modality_dir}. Run the data generator first."
        )

    X, y = [], []
    for label, sid in enumerate(subjects):
        for f in sorted((modality_dir / sid).glob("*.png")):
            X.append(preprocess(f))
            y.append(label)

    return np.stack(X), np.asarray(y, dtype="int64"), subjects


def train_val_split(X, y, val_fraction: float, seed: int):
    """Stratified-ish split: hold out a fixed fraction of each class for validation."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
