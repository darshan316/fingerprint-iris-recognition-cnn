"""
CNN definition shared by both modalities.

It is a compact VGG-style stack. The important design choice is the named
"embedding" layer just before the softmax head: training optimises the whole
thing as an identity classifier, but at match time we throw the softmax away
and read that embedding instead. Cross-entropy pulls same-subject embeddings
together and pushes different subjects apart, which is exactly the geometry a
cosine-similarity matcher wants.

We deliberately use only stock Keras layers so a saved model reloads with
keras.models.load_model() and no custom_objects bookkeeping.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import CHANNELS, EMBEDDING_DIM, IMG_SIZE

EMBEDDING_LAYER = "embedding"

# We train on a small number of subjects with few steps per epoch, so the
# default BatchNorm momentum (0.99) never lets the running mean/variance catch
# up — train accuracy looks perfect while inference-time accuracy sits at
# chance. A faster-adapting momentum fixes that completely.
BN_MOMENTUM = 0.85


def _conv_block(x, filters, drop, repeats=2):
    for _ in range(repeats):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(drop)(x)
    return x


def build_cnn(num_classes: int,
              img_size: int = IMG_SIZE,
              channels: int = CHANNELS,
              embedding_dim: int = EMBEDDING_DIM) -> keras.Model:
    inp = keras.Input(shape=(img_size, img_size, channels), name="image")

    x = _conv_block(inp, 32, 0.10)
    x = _conv_block(x, 64, 0.15)
    x = _conv_block(x, 128, 0.20, repeats=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(embedding_dim, name=EMBEDDING_LAYER)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.Activation("relu", name="embedding_act")(x)
    out = layers.Dense(num_classes, activation="softmax", name="identity")(x)

    return keras.Model(inp, out, name="biometric_cnn")


def embedding_model(trained: keras.Model) -> keras.Model:
    """View of a trained classifier that outputs the embedding vector."""
    layer = trained.get_layer("embedding_act")
    return keras.Model(trained.input, layer.output, name=f"{trained.name}_embed")
