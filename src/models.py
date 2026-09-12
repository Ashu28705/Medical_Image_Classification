"""Baseline CNN and ResNet50 model definitions."""

from __future__ import annotations

import os
from pathlib import Path

# The local host maps Keras' default temp cache to a non-writable location.
# A project-local cache also makes downloaded pretrained weights auditable.
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras_cache"))

import tensorflow as tf


def build_baseline_cnn(image_size: int = 224, dropout_rate: float = 0.35) -> tf.keras.Model:
    """A small CNN baseline; augmentation is active only during model.fit training mode."""
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
        tf.keras.layers.RandomContrast(0.08),
    ], name="training_augmentation")
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="xray_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255, name="rescale_to_unit_range")(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_resnet50(image_size: int = 224, dropout_rate: float = 0.35, weights: str | None = "imagenet") -> tf.keras.Model:
    """ResNet50 transfer-learning model; its base starts frozen."""
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="xray_image")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=x)
    base.trainable = False
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    return tf.keras.Model(inputs, outputs, name="resnet50_transfer_learning")


def unfreeze_resnet_top(model: tf.keras.Model, trainable_layers: int = 30) -> None:
    """Unfreeze the last convolutional layers in the saved flat functional model.

    Keras saves the ResNet50 backbone as a flat graph when using the functional API,
    so the base is not represented as a nested `resnet50` submodel after reload.
    We therefore unfreeze the final convolutional block layers directly from the
    top-level model graph while keeping BatchNorm frozen in the trainable section.
    """
    candidate_layers = [
        layer for layer in model.layers
        if not isinstance(layer, (
            tf.keras.layers.InputLayer,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.Dropout,
            tf.keras.layers.Dense,
        ))
    ]
    if not candidate_layers:
        return

    for layer in model.layers:
        layer.trainable = False

    for layer in candidate_layers[-trainable_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
