"""Grad-CAM for a saved Keras binary classifier; it is interpretability, not clinical proof."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_xray(image_path: str | Path, image_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Return displayable RGB image and batch-ready float image."""
    with Image.open(image_path) as image:
        display_image = np.asarray(image.convert("RGB").resize((image_size, image_size)))
    return display_image, np.expand_dims(display_image.astype("float32"), axis=0)


def find_last_convolution_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the final Conv2D layer, including within a nested transfer-learning base."""
    candidates: list[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            candidates.extend(item for item in layer.layers if isinstance(item, tf.keras.layers.Conv2D))
        elif isinstance(layer, tf.keras.layers.Conv2D):
            candidates.append(layer)
    if not candidates:
        raise ValueError("No Conv2D layer was found for Grad-CAM.")
    return candidates[-1]


def make_gradcam_heatmap(image_batch: np.ndarray, model: tf.keras.Model) -> tuple[np.ndarray, float, str]:
    last_conv = find_last_convolution_layer(model)
    gradient_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        convolution_outputs, predictions = gradient_model(image_batch)
        class_score = predictions[:, 0]
    gradients = tape.gradient(class_score, convolution_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    convolution_outputs = convolution_outputs[0]
    heatmap = convolution_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), float(predictions[0, 0]), last_conv.name


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    """Make color heatmap and overlay, both in uint8 RGB."""
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], original.shape[:2]).numpy().squeeze()
    color_heatmap = tf.cast(tf.round(255 * tf.image.resize(heatmap_resized[..., np.newaxis], original.shape[:2])), tf.uint8)
    color_heatmap = tf.image.grayscale_to_rgb(color_heatmap)
    # A lightweight red/yellow color mapping avoids an additional OpenCV dependency.
    values = heatmap_resized
    colored = np.stack([np.full_like(values, 255), (255 * (1 - values)).astype(np.uint8), np.zeros_like(values)], axis=-1)
    colored = (colored * values[..., np.newaxis]).astype(np.uint8)
    overlay = np.clip((1 - alpha) * original + alpha * colored, 0, 255).astype(np.uint8)
    return colored, overlay
