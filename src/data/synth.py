"""
Synthetic biometric generation.

Real fingerprint / iris corpora (SOCOFing, CASIA, MMU ...) need licences and
gigabytes of downloads, which makes a project awkward to just clone-and-run.
Instead we fabricate a small closed-world population that behaves the way the
real thing does for our purposes: every subject owns a *unique* pattern, and
every capture of that subject is a noisy, slightly transformed view of it.
That intra-subject variation is exactly what the CNN has to see through.

Fingerprints are built from an oriented ridge flow with a singular (core)
point, the way a loop/whorl print actually looks. Irises are synthesised in
the polar "rubber-sheet" coordinates Daugman uses, then warped back to the
image plane, so pupil dilation between captures stretches the texture the way
a real eye does.

Nothing here is meant to fool a forensic examiner. It is meant to give a
recognition pipeline something honest to learn.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from config import IMG_SIZE, SEED

# A friendly roster so results read like "verified: Priya Nair" instead of an
# opaque index. Index i maps to subject id SUBJ_<i>.
ROSTER = [
    "Aarav Sharma", "Priya Nair", "Daniel Okafor", "Mei Lin",
    "Sofia Rossi", "Omar Haddad", "Hannah Schmidt", "Ravi Patel",
    "Yuki Tanaka", "Grace Adeyemi", "Lucas Moreau", "Ana Costa",
    "Ethan Brooks", "Fatima Zahra", "Noah Williams", "Ingrid Larsen",
]


def subject_id(index: int) -> str:
    return f"SUBJ_{index:02d}"


def subject_name(index: int) -> str:
    return ROSTER[index % len(ROSTER)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _seed_for(subject_index: int, modality: str, sample: int | None) -> int:
    """Deterministic seed so a given (subject, modality, sample) is reproducible."""
    tag = 0 if modality == "fingerprint" else 1
    s = sample if sample is not None else 9999
    return int((SEED * 1_000_003 + subject_index * 7919 + tag * 104729 + s * 131) % (2**31 - 1))


def _smooth_noise(shape, rng, blur_radius=3.0):
    """Low-frequency random field in [0, 1] (used for crypts / smudges)."""
    n = rng.random(shape).astype("float32")
    im = Image.fromarray((n * 255).astype("uint8"))
    im = im.filter(ImageFilter.GaussianBlur(blur_radius))
    out = np.asarray(im, dtype="float32") / 255.0
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def _to_uint8(arr):
    return np.clip(arr * 255.0, 0, 255).astype("uint8")


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------
def synth_fingerprint(subject_index: int, sample: int | None = None,
                      size: int = IMG_SIZE) -> np.ndarray:
    """Render one grayscale fingerprint capture for a subject.

    Identity-fixed traits (ridge frequency, the core location, the swirl that
    makes a loop or a whorl, the global ridge direction) are drawn from a seed
    that depends only on the subject. Per-capture nuisances (rotation, shift,
    contrast, noise) are redrawn every sample. So two prints of one subject
    differ, but far less than two prints of different subjects.
    """
    fixed = np.random.default_rng(_seed_for(subject_index, "fingerprint", None))
    cap = np.random.default_rng(_seed_for(subject_index, "fingerprint", sample))

    # --- traits that define this finger -----------------------------------
    base_freq = fixed.uniform(9.0, 13.5)            # ridge cycles across the print
    core = fixed.uniform(-0.30, 0.30, size=2)       # singular point
    swirl = fixed.uniform(0.7, 1.7) * fixed.choice([-1.0, 1.0])
    ridge_dir = fixed.uniform(0.0, np.pi)
    warp_amp = fixed.uniform(0.05, 0.11)
    warp_freq = fixed.uniform(1.2, 2.8, size=4)
    warp_phase = fixed.uniform(0.0, 2 * np.pi, size=4)

    # --- how this particular capture was taken ----------------------------
    rot = cap.uniform(-0.13, 0.13)
    shift = cap.uniform(-0.06, 0.06, size=2)
    freq_jit = cap.uniform(0.96, 1.04)
    noise_sigma = cap.uniform(0.05, 0.13)

    lin = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(lin, lin)
    X, Y = X - shift[0], Y - shift[1]
    Xr = X * np.cos(rot) - Y * np.sin(rot)
    Yr = X * np.sin(rot) + Y * np.cos(rot)

    # swirl about the core point -> a loop/whorl-like singularity
    dx, dy = Xr - core[0], Yr - core[1]
    r = np.hypot(dx, dy) + 1e-6
    ang = np.arctan2(dy, dx) + swirl * np.exp(-(r ** 2) / 0.5)
    Xs = core[0] + r * np.cos(ang)
    Ys = core[1] + r * np.sin(ang)

    # smooth bending of the ridges
    warp = np.zeros_like(Xs)
    for k in range(4):
        coord = Xs if k % 2 == 0 else Ys
        warp += warp_amp * np.sin(warp_freq[k] * np.pi * coord + warp_phase[k])

    proj = Xs * np.cos(ridge_dir) + Ys * np.sin(ridge_dir) + warp
    ridges = np.sin(2 * np.pi * base_freq * freq_jit * proj)

    ridges = 0.5 * (ridges + 1.0)                   # -> 0..1
    ridges = np.clip((ridges - 0.45) * 3.6, 0, 1)   # sharpen crests into ridges
    img = 1.0 - ridges                              # ridges dark, valleys light

    # worn patches / pressure variation
    smudge = _smooth_noise((size, size), cap, blur_radius=size / 12.0)
    img = img * (0.7 + 0.3 * smudge)
    img = img + cap.normal(0.0, noise_sigma, img.shape)

    # the print only covers an elliptical area; the rest is sensor background
    ex = (X / 0.92) ** 2 + (Y / 0.86) ** 2
    mask = np.clip(1.2 - ex, 0, 1) ** 0.7
    bg = 0.93 - 0.05 * _smooth_noise((size, size), fixed, blur_radius=size / 6.0)
    img = img * mask + bg * (1 - mask)

    out = Image.fromarray(_to_uint8(img)).filter(ImageFilter.GaussianBlur(0.5))
    return np.asarray(out, dtype="uint8")


# --------------------------------------------------------------------------
# iris
# --------------------------------------------------------------------------
def synth_iris(subject_index: int, sample: int | None = None,
               size: int = IMG_SIZE) -> np.ndarray:
    """Render one grayscale iris capture for a subject.

    The texture is composed in polar coordinates (radial furrows + angular
    structure + fine crypts), which is where iris information actually lives.
    Pupil dilation, eye tilt and illumination change per capture; the texture
    itself is fixed to the subject.
    """
    fixed = np.random.default_rng(_seed_for(subject_index, "iris", None))
    cap = np.random.default_rng(_seed_for(subject_index, "iris", sample))

    # --- the iris that belongs to this subject ----------------------------
    n_radial = int(fixed.integers(4, 8))
    n_angular = int(fixed.integers(5, 11))
    w_radial = fixed.normal(0, 1, n_radial)
    w_angular = fixed.normal(0, 1, n_angular)
    ph_radial = fixed.uniform(0, 2 * np.pi, n_radial)
    ph_angular = fixed.uniform(0, 2 * np.pi, n_angular)
    base_gray = fixed.uniform(0.34, 0.60)
    iris_r = fixed.uniform(0.62, 0.74)
    crypts = _smooth_noise((size, size), fixed, blur_radius=size / 22.0)

    # --- this capture ------------------------------------------------------
    pupil_r = cap.uniform(0.22, 0.40) * iris_r       # dilation
    rot = cap.uniform(-0.28, 0.28)
    shift = cap.uniform(-0.04, 0.04, size=2)
    illum = cap.uniform(0.85, 1.15)
    noise_sigma = cap.uniform(0.02, 0.05)
    has_eyelid = cap.random() < 0.45

    lin = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(lin, lin)
    X, Y = X - shift[0], Y - shift[1]
    Xr = X * np.cos(rot) - Y * np.sin(rot)
    Yr = X * np.sin(rot) + Y * np.cos(rot)
    r = np.hypot(Xr, Yr)
    theta = np.arctan2(Yr, Xr)

    # normalised radius across the iris ring (Daugman rubber sheet)
    rho = np.clip((r - pupil_r) / (iris_r - pupil_r + 1e-6), 0, 1)
    tex = np.zeros_like(r)
    for k in range(n_radial):
        tex += w_radial[k] * np.sin((k + 1) * np.pi * rho + ph_radial[k])
    for k in range(n_angular):
        tex += w_angular[k] * np.sin((k + 1) * theta + ph_angular[k])
    tex = tex / (n_radial + n_angular) ** 0.5
    tex = tex + 1.4 * (crypts - 0.5)
    iris_val = np.clip(base_gray + 0.22 * tex * illum, 0.05, 0.95)

    img = np.full((size, size), 0.92, dtype="float32")        # sclera
    ring = (r >= pupil_r) & (r <= iris_r)
    img[ring] = iris_val[ring]

    # limbus: darken the outer iris boundary a touch
    limbus = (r > iris_r - 0.05) & (r <= iris_r)
    img[limbus] *= 0.7

    pupil = r < pupil_r
    img[pupil] = 0.03
    # specular highlight, the little catch-light every iris camera leaves
    hl = np.hypot(Xr - 0.10, Yr + 0.10) < 0.06
    img[hl & pupil] = 0.9

    if has_eyelid:
        lid = cap.uniform(0.55, 0.8)
        img[Y < -lid] = 0.85
        img[Y > lid] = 0.85

    img = img + cap.normal(0, noise_sigma, img.shape)
    out = Image.fromarray(_to_uint8(img)).filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(out, dtype="uint8")


# --------------------------------------------------------------------------
# population generation
# --------------------------------------------------------------------------
def generate_population(num_subjects: int, samples_per_subject: int,
                        finger_dir, iris_dir, size: int = IMG_SIZE,
                        progress=None) -> None:
    """Render the whole synthetic population to disk.

    Layout: <finger_dir>/SUBJ_03/cap_07.png and the matching iris under
    <iris_dir>/SUBJ_03/cap_07.png. Same folder name == same person across the
    two modalities, which is what enrollment relies on.
    """
    from pathlib import Path

    finger_dir, iris_dir = Path(finger_dir), Path(iris_dir)
    total = num_subjects * samples_per_subject
    done = 0
    for s in range(num_subjects):
        sid = subject_id(s)
        (finger_dir / sid).mkdir(parents=True, exist_ok=True)
        (iris_dir / sid).mkdir(parents=True, exist_ok=True)
        for c in range(samples_per_subject):
            fp = synth_fingerprint(s, sample=c, size=size)
            ir = synth_iris(s, sample=c, size=size)
            Image.fromarray(fp).save(finger_dir / sid / f"cap_{c:02d}.png")
            Image.fromarray(ir).save(iris_dir / sid / f"cap_{c:02d}.png")
            done += 1
            if progress and done % 20 == 0:
                progress(done, total)
    if progress:
        progress(total, total)
