"""Inference helpers for the independently trained TB ResNet50 model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from gradcam import load_xray, make_gradcam_heatmap, overlay_heatmap

EXPECTED_CLASS_TO_INDEX = {"NORMAL": 0, "TUBERCULOSIS": 1}
EXPECTED_PREPROCESS_FUNCTION = "tf.keras.applications.resnet50.preprocess_input"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_tuberculosis_runtime(
    model_path: str | Path,
    label_mapping_path: str | Path,
    preprocessing_config_path: str | Path,
) -> tuple[tf.keras.Model, dict[str, Any], dict[str, Any]]:
    model_path = Path(model_path)
    mapping_path = Path(label_mapping_path)
    preprocessing_path = Path(preprocessing_config_path)
    for path in (model_path, mapping_path, preprocessing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Tuberculosis artifact not found: {path}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    class_to_index = mapping.get("class_to_index")
    if class_to_index != EXPECTED_CLASS_TO_INDEX:
        raise ValueError(f"Unrecognized Tuberculosis class mapping: {class_to_index}")
    if mapping.get("positive_class") != "TUBERCULOSIS":
        raise ValueError("Tuberculosis model metadata must declare TUBERCULOSIS as positive")
    if preprocessing.get("input_color_space") != "RGB":
        raise ValueError("Unsupported Tuberculosis input color space")
    if preprocessing.get("input_pixel_range_before_preprocess") != [0, 255]:
        raise ValueError("Tuberculosis model expects RGB image values in [0, 255]")
    resize = preprocessing.get("resize")
    if not isinstance(resize, list) or len(resize) != 2 or not all(
        isinstance(dimension, int) and dimension >= 32 for dimension in resize
    ):
        raise ValueError(f"Invalid Tuberculosis image resize configuration: {resize}")
    if preprocessing.get("preprocess_function") != EXPECTED_PREPROCESS_FUNCTION:
        raise ValueError("Unsupported Tuberculosis ResNet50 preprocessing configuration")
    threshold = preprocessing.get("prediction_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError(f"Invalid prediction threshold in {preprocessing_path}")

    model = tf.keras.models.load_model(model_path)
    expected_input_shape = (None, resize[0], resize[1], 3)
    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(
            f"Tuberculosis model input {model.input_shape} does not match saved preprocessing {expected_input_shape}"
        )
    if tuple(model.output_shape) != (None, 1):
        raise ValueError(f"Tuberculosis model must have one sigmoid output, got {model.output_shape}")
    output_layer = model.layers[-1]
    activation = getattr(getattr(output_layer, "activation", None), "__name__", None)
    if activation != "sigmoid":
        raise ValueError(f"Tuberculosis output layer is not sigmoid: {activation}")
    return model, mapping, preprocessing


def predict_tuberculosis(
    model: tf.keras.Model,
    image_path: str | Path,
    label_mapping: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    if label_mapping.get("class_to_index") != EXPECTED_CLASS_TO_INDEX:
        raise ValueError("Tuberculosis output mapping has not been verified")
    image_path = Path(image_path)
    if image_path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image format: {image_path.suffix or '(no extension)'}")
    if preprocessing_config.get("preprocess_function") != EXPECTED_PREPROCESS_FUNCTION:
        raise ValueError("Unsupported Tuberculosis preprocessing configuration")
    height, width = preprocessing_config["resize"]
    try:
        original, batch = load_xray(image_path, image_size=height)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Could not read chest X-ray image {image_path.name}: {error}") from error
    if width != height:
        with Image.open(image_path) as image:
            original = np.asarray(image.convert("RGB").resize((width, height)))
            batch = np.expand_dims(original.astype(np.float32), axis=0)

    prediction = np.asarray(model.predict(batch, verbose=0), dtype=np.float64).reshape(-1)
    if prediction.shape != (1,) or not np.isfinite(prediction).all():
        raise ValueError(f"Tuberculosis model returned invalid probabilities: {prediction}")
    positive_probability = float(prediction[0])
    if not 0 <= positive_probability <= 1:
        raise ValueError(f"Tuberculosis probability is outside [0, 1]: {positive_probability}")
    probabilities_by_index = {0: 1.0 - positive_probability, 1: positive_probability}
    index_to_class = {index: name for name, index in label_mapping["class_to_index"].items()}
    probabilities = {
        index_to_class[index]: probability
        for index, probability in probabilities_by_index.items()
    }
    threshold = float(preprocessing_config["prediction_threshold"])
    predicted_index = 1 if positive_probability >= threshold else 0

    heatmap, gradcam_probability, layer_name = make_gradcam_heatmap(batch, model)
    if not np.isfinite(heatmap).all() or heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError("Tuberculosis Grad-CAM returned an invalid heatmap")
    color_heatmap, overlay = overlay_heatmap(original, heatmap)
    return {
        "disease": "Tuberculosis",
        "predicted_class": index_to_class[predicted_index],
        "confidence": float(probabilities[index_to_class[predicted_index]]),
        "probabilities": probabilities,
        "original": original,
        "heatmap": color_heatmap,
        "overlay": overlay,
        "gradcam_layer": layer_name,
        "gradcam_probability": float(gradcam_probability),
    }