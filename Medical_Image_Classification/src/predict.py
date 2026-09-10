"""Reusable prediction API for a saved pneumonia-classification research model."""

from __future__ import annotations

from pathlib import Path

import typing
try:
    import tensorflow as tf
    from gradcam import load_xray, make_gradcam_heatmap, overlay_heatmap
except ImportError:
    tf = None



def predict_xray(model: typing.Any, image_path: str | Path, image_size: int = 224) -> dict:
    original, batch = load_xray(image_path, image_size)
    heatmap, pneumonia_probability, layer_name = make_gradcam_heatmap(batch, model)
    color_heatmap, overlay = overlay_heatmap(original, heatmap)
    normal_probability = 1 - pneumonia_probability
    predicted_label = "PNEUMONIA" if pneumonia_probability >= 0.5 else "NORMAL"
    confidence = pneumonia_probability if predicted_label == "PNEUMONIA" else normal_probability
    return {
        "predicted_class": predicted_label,
        "confidence": float(confidence),
        "probabilities": {"NORMAL": float(normal_probability), "PNEUMONIA": float(pneumonia_probability)},
        "original": original, "heatmap": color_heatmap, "overlay": overlay,
        "gradcam_layer": layer_name,
    }
