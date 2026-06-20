# BioVerify — Fingerprint & Iris Recognition with CNNs

A multimodal biometric verification system. You give it a fingerprint image
and an iris image and it answers one question: do these two belong to the
same enrolled person? Two separate convolutional networks turn the images into
feature vectors, those vectors are matched against a stored gallery of enrolled
subjects and the two scores are fused into a single decision.

Everything runs locally and out of the box  there is no external dataset to
download. A synthetic population of subjects is generated on first run, so you
can train, enrol and demo the whole thing in a couple of minutes.

## What it actually does

When you press Verify, the app:

1. Runs the fingerprint through the fingerprint CNN and the iris through the
   iris CNN, producing a 128-d embedding for each.
2. Compares each embedding (cosine similarity) against every stored template in
   the enrollment gallery, giving a best-matching subject + score per modality.
3. Fuses the two similarity vectors with a weighted sum (iris weighted slightly
   higher, as it carries more discriminative detail in practice).
4. Returns one of three verdicts:

| Verdict | Meaning |
|---------|---------|
| Same person | Both modalities point at the same enrolled subject and the fused confidence clears the threshold. |
| Different people | Both are recognised, but the fingerprint and the iris belong to different enrolled subjects. |
| Not recognised | At least one sample can't be confidently placed in the gallery (not enrolled, or a poor capture). |

## How it works

Synthetic biometrics (src/data/synth.py). Real fingerprint/iris corpora
need licences and large downloads, so the project fabricates its own. Each
subject is given a unique pattern and several noisy captures of it are
rendered so the network has within-subject variation to learn through:

 Fingerprints are built from an oriented ridge flow with a singular (core)
  point  the loop/whorl structure a real print has  using a warped Gabor-style
  sinusoid. Per capture, the finger is randomly rotated, shifted and degraded.
 Irises are synthesised in the polar rubber-sheet coordinates Daugman uses
  (radial furrows + angular structure + fine crypts), then mapped back to the
  image plane. Pupil dilation, eye tilt and illumination vary per capture.

The CNNs (src/models/cnn.py). A compact VGG-style stack shared by both
modalities. The layer just before the softmax head is a named embedding
layer, training optimises an identity classifier, but at match time we discard
the softmax and read that embedding. Cross-entropy pulls same-subject embeddings
together and pushes different subjects apart  the geometry a cosine matcher
wants. (Note: BatchNorm momentum is lowered to 0.85, with few steps per epoch
the default 0.99 never lets the running statistics converge, which silently
wrecks inference accuracy.)

Enrollment (src/enrollment.py). Every capture of a subject is embedded and
averaged into one L2-normalised template per modality. These templates are the
stored data the matcher compares against, saved to enrollment/templates.npz.

Matching & fusion (src/matcher.py). Cosine similarity to the gallery,
score-level fusion and the three-way decision above. The thresholds in
config.py were calibrated from the measured genuine-vs-impostor score
distributions rather than guessed (see scripts/evaluate.py).

## Project structure


fingerprint_iris_recognition/
├── main.py                  # launches the GUI
├── config.py                # all tunables: sizes, paths, thresholds
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── synth.py          # synthetic fingerprint + iris generators
│   │   └── dataset.py        # image loading + preprocessing
│   ├── models/
│   │   └── cnn.py            # the CNN architecture
│   ├── train.py              # per-modality training
│   ├── enrollment.py         # builds the template gallery
│   └── matcher.py            # cosine matching + fusion + decision
├── gui/
│   ├── app.py                # CustomTkinter desktop app
│   └── theme.py              # colours / fonts
├── scripts/
│   ├── generate_data.py      # render the synthetic population
│   ├── train_models.py       # train both CNNs
│   ├── build_enrollment.py   # build the gallery
│   ├── evaluate.py           # genuine / cross / stranger evaluation
│   └── run_all.py            # do everything end-to-end
├── data/        (generated)  # synthetic captures
├── models/      (generated)  # trained .keras files + label maps
├── enrollment/  (generated)  # templates.npz gallery
└── samples/     (generated)  # ready-made probe images for the GUI


## Quickstart

Requires Python 3.9–3.11.

bash
# 1. install dependencies
pip install -r requirements.txt

# 2. generate data, train both CNNs and build the gallery (~1–2 min, CPU is fine)
python scripts/run_all.py

# 3. launch the app
python main.py


run_all.py prints an evaluation at the end  expect ~100% accuracy
distinguishing genuine pairs from cross pairs on the synthetic data.

### Using the GUI

Click Upload image under Fingerprint and Iris (or just hit Load demo
pair to drop in a ready-made genuine pair from samples/), then press
Verify identity. The result card turns green / red / amber for
same-person / different / not-recognised, shows a fused confidence bar and
breaks down which subject each modality matched.

To try the other cases, load samples/mismatch_.png (a fingerprint and iris
from two different subjects) or samples/stranger_.png (someone never
enrolled).

## Configuration

Edit config.py  the population size, image size, training length, fusion
weights and decision thresholds all live there. Any size knob can also be set
from the environment for quick experiments, e.g.:

bash
BIO_SUBJECTS=20 BIO_EPOCHS=30 python scripts/run_all.py


Re-run scripts/run_all.py after changing anything that affects the data or the
models.

## Honest limitations

 The biometric images are synthetic. The pipeline (CNN → embedding →
  fusion → decision) is the real deal, but the inputs are generated, so this is
  a working demonstrator, not a deployable security product. To use real data,
  point src/data/dataset.load_modality at a folder of subject/capture.png
  images (e.g. SOCOFing for prints, MMU/CASIA for irises) and skip the generator.
 Recognising that a sample is not enrolled at all (open-set rejection) is
  fundamentally harder than telling enrolled subjects apart  a closed-set
  classifier will always map a stranger somewhere. The thresholds catch most
  strangers and, importantly, never grant a same person verdict to one, but
  some land in different people. This is an inherent trade-off, not a bug.
 It is a verification demo. Don't use it to make decisions about real people.

## Requirements

TensorFlow / Keras (the CNNs), Pillow + NumPy (image generation and math) and
CustomTkinter (the GUI). Pinned in requirements.txt.
