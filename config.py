"""
Central configuration for the multimodal biometric system.

Everything tunable lives here so the data generator, the training scripts,
the GUI and the matcher all agree on image sizes, paths and thresholds.
Edit the values, not the code that reads them.
"""

import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    """Allow any of the size knobs to be overridden from the environment,
    e.g. BIO_EPOCHS=5 python scripts/train_models.py for a quick dry run."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- Project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"                 # synthetic images live here
FINGER_DIR = DATA_DIR / "fingerprint"
IRIS_DIR = DATA_DIR / "iris"

MODEL_DIR = ROOT / "models"              # trained .keras files + label maps
ENROLL_DIR = ROOT / "enrollment"         # the gallery of stored templates
ENROLL_DB = ENROLL_DIR / "templates.npz"

for _d in (DATA_DIR, FINGER_DIR, IRIS_DIR, MODEL_DIR, ENROLL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Synthetic population -------------------------------------------------
# We fabricate a small closed-world population. Each "subject" gets a unique
# fingerprint pattern and a unique iris texture, then we render several noisy
# captures of each so the network has within-subject variation to learn from.
NUM_SUBJECTS = _int_env("BIO_SUBJECTS", 12)
SAMPLES_PER_SUBJECT = _int_env("BIO_SAMPLES", 22)
SUBJECT_PREFIX = "SUBJ"                   # ids look like SUBJ_03

# --- Image geometry -------------------------------------------------------
IMG_SIZE = _int_env("BIO_IMG_SIZE", 64)   # square, grayscale, fed to both CNNs
CHANNELS = 1

# --- CNN / training -------------------------------------------------------
EMBEDDING_DIM = 128                       # length of the template vector
EPOCHS = _int_env("BIO_EPOCHS", 20)
BATCH_SIZE = _int_env("BIO_BATCH", 32)
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.18

# --- Matching & fusion ----------------------------------------------------
# Each modality returns a cosine similarity in [0, 1] against the gallery.
# We fuse them with a weighted sum; iris is weighted a little higher because
# in practice iris carries more discriminative entropy than a single print.
FUSION_WEIGHTS = {"fingerprint": 0.45, "iris": 0.55}

# A modality only "recognises" a subject if its own best score clears this.
# These two were calibrated from the genuine vs. impostor score distributions
# (see scripts/evaluate.py): genuine matches score ~0.94+ per modality and
# ~0.96+ fused, while unenrolled "strangers" rarely exceed ~0.86 fused. The
# gap is what lets us separate same-person from not-in-gallery.
PER_MODALITY_THRESHOLD = 0.84
# The fused score must clear this for an overall accept.
FUSION_THRESHOLD = 0.90

# Reproducibility. Change it if you want a different synthetic population.
SEED = 1234
