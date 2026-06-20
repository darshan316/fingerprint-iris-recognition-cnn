"""
Training for a single modality.

Each modality (fingerprint, iris) gets its own CNN trained to tell the
enrolled subjects apart. We train on captures, hold out a fraction for
validation, and lightly augment so the network learns the pattern rather
than the exact pixel placement of one capture.

Outputs per modality, written to MODEL_DIR:
  <modality>.keras          the trained classifier
  <modality>_labels.json    label-index -> subject id, plus a little metadata
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.data.dataset import load_modality, train_val_split
from src.models.cnn import build_cnn

MODALITY_DIRS = {
    "fingerprint": config.FINGER_DIR,
    "iris": config.IRIS_DIR,
}


def _augmenter() -> keras.Sequential:
    # Mild geometric jitter only — enough to stop the net memorising exact
    # placement, not so much that a print/iris stops being itself.
    return keras.Sequential(
        [
            layers.RandomRotation(0.04),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomZoom(0.08),
        ],
        name="augment",
    )


def _make_dataset(X, y, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=config.SEED)
    ds = ds.batch(config.BATCH_SIZE)
    if training:
        aug = _augmenter()
        ds = ds.map(lambda a, b: (aug(a, training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_modality(modality: str, verbose: int = 1) -> dict:
    if modality not in MODALITY_DIRS:
        raise ValueError(f"unknown modality {modality!r}")

    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    X, y, subjects = load_modality(MODALITY_DIRS[modality])
    Xtr, ytr, Xval, yval = train_val_split(X, y, config.VAL_FRACTION, config.SEED)

    model = build_cnn(num_classes=len(subjects))
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
    ]

    history = model.fit(
        _make_dataset(Xtr, ytr, training=True),
        validation_data=_make_dataset(Xval, yval, training=False),
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=verbose,
    )

    val_loss, val_acc = model.evaluate(_make_dataset(Xval, yval, False), verbose=0)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(config.MODEL_DIR / f"{modality}.keras"))
    with open(config.MODEL_DIR / f"{modality}_labels.json", "w") as fh:
        json.dump(
            {
                "modality": modality,
                "labels": subjects,
                "img_size": config.IMG_SIZE,
                "embedding_dim": config.EMBEDDING_DIM,
                "val_accuracy": round(float(val_acc), 4),
            },
            fh,
            indent=2,
        )

    return {
        "modality": modality,
        "subjects": len(subjects),
        "train": len(Xtr),
        "val": len(Xval),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs_run": len(history.history["loss"]),
    }


def train_all(verbose: int = 1) -> list[dict]:
    results = []
    for modality in MODALITY_DIRS:
        if verbose:
            print(f"\n=== training {modality} CNN ===")
        results.append(train_modality(modality, verbose=verbose))
    return results
