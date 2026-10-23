"""MEDVISION AI: Streamlit research-prototype interface for a real saved Keras model."""

from __future__ import annotations

import sys
import tempfile
import hashlib
import json
from io import BytesIO
from pathlib import Path

import streamlit as st
import typing
try:
    import tensorflow as tf
except ImportError:
    tf = None
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from predict import predict_xray  # noqa: E402
if tf is not None:
    from predict_tuberculosis import load_tuberculosis_runtime, predict_tuberculosis  # noqa: E402
else:
    load_tuberculosis_runtime = None
    predict_tuberculosis = None

PNEUMONIA_MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "final_model.keras"
PNEUMONIA_CONFIG_PATH = PROJECT_ROOT / "results" / "resnet50" / "frozen" / "run_config.json"
TUBERCULOSIS_MODEL_DIR = PROJECT_ROOT / "models" / "tuberculosis"
TUBERCULOSIS_MODEL_PATH = TUBERCULOSIS_MODEL_DIR / "best_model.keras"
TUBERCULOSIS_MAPPING_PATH = TUBERCULOSIS_MODEL_DIR / "label_mapping.json"
TUBERCULOSIS_PREPROCESSING_PATH = TUBERCULOSIS_MODEL_DIR / "preprocessing_config.json"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

st.set_page_config(page_title="MEDVISION AI", page_icon="◈", layout="wide")
st.markdown("""
<style>
  .stApp { background: #081116; color: #dce9eb; }
  .block-container { max-width: 1280px; padding-top: 2.2rem; }
  .hero { border-left: 3px solid #49c4cf; padding: .2rem 0 .2rem 1rem; margin-bottom: 1.4rem; }
  .hero h1 { font-family: monospace; font-size: 2rem; letter-spacing: .12em; margin: 0; color: #f0f9fa; }
  .hero p { color: #8ba5aa; margin: .35rem 0 0; font-family: monospace; }
  .panel { border: 1px solid #1e3b42; background: #0c1a20; border-radius: 8px; padding: 1.2rem; min-height: 140px; }
  .section-label { color: #49c4cf; font-family: monospace; letter-spacing: .12em; font-size: .8rem; }
  .readout { font-family: monospace; font-size: 1.7rem; color: #f1c779; letter-spacing: .06em; }
  .disclaimer { border: 1px solid #5f573a; background: #17150e; color: #e2d8ac; padding: .8rem 1rem; border-radius: 6px; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pneumonia_model(path: str, config_path: str) -> tuple[typing.Any, dict]:
    if tf is None:
        raise RuntimeError("TensorFlow could not initialize")
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Pneumonia label/config file not found: {config_file}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("classes") != {"NORMAL": 0, "PNEUMONIA": 1}:
        raise ValueError("Pneumonia model class mapping is missing or incompatible")
    model_file = Path(path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Pneumonia model artifact not found: {model_file}")
    model = tf.keras.models.load_model(model_file)
    image_size = int(config.get("image_size", 0))
    if tuple(model.input_shape) != (None, image_size, image_size, 3):
        raise ValueError(f"Pneumonia model input shape does not match config: {model.input_shape}")
    if tuple(model.output_shape) != (None, 1):
        raise ValueError(f"Pneumonia model output shape is incompatible: {model.output_shape}")
    if getattr(getattr(model.layers[-1], "activation", None), "__name__", None) != "sigmoid":
        raise ValueError("Pneumonia model output is not a sigmoid probability")
    return model, config


@st.cache_resource(show_spinner=False)
def load_tuberculosis_model(path: str, mapping_path: str, preprocessing_path: str):
    if tf is None or load_tuberculosis_runtime is None:
        raise RuntimeError("TensorFlow or the Tuberculosis inference module could not initialize")
    return load_tuberculosis_runtime(path, mapping_path, preprocessing_path)


st.markdown("""
<div class="hero"><h1>MEDVISION AI</h1><p>DEEP LEARNING-BASED CHEST X-RAY IMAGE CLASSIFICATION · RESEARCH CONSOLE</p></div>
""", unsafe_allow_html=True)

left, right = st.columns((1, 1.25), gap="large")
with left:
    st.markdown('<div class="section-label">01 / IMAGE INTAKE</div>', unsafe_allow_html=True)
    selected_disease = st.selectbox("Disease", ("Pneumonia", "Tuberculosis"), key="selected_disease")
    uploaded_file = st.file_uploader("Upload a compatible chest X-ray image", type=["jpg", "jpeg", "png"])
    upload_bytes = uploaded_file.getvalue() if uploaded_file else None
    upload_digest = hashlib.sha256(upload_bytes).hexdigest() if upload_bytes is not None else None
    result_state = st.session_state.get("result")
    if result_state and (
        result_state.get("disease") != selected_disease
        or result_state.get("upload_digest") != upload_digest
    ):
        st.session_state.pop("result", None)
        result_state = None

    upload_is_valid = False
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix.casefold()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            st.error("Unsupported image format. Upload a JPG, JPEG, or PNG chest X-ray.")
        else:
            try:
                with Image.open(BytesIO(upload_bytes)) as image:
                    image.verify()
                with Image.open(BytesIO(upload_bytes)) as image:
                    preview = image.convert("RGB")
                upload_is_valid = True
                st.image(preview, caption="Uploaded image preview", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                st.error(f"The uploaded file is not a readable image: {error}")

if selected_disease == "Pneumonia":
    selected_model_path = PNEUMONIA_MODEL_PATH
    model_label = "ResNet50 / Pneumonia"
else:
    selected_model_path = TUBERCULOSIS_MODEL_PATH
    model_label = "ResNet50 / Tuberculosis"

selected_model = None
selected_mapping = None
selected_preprocessing = None
model_error = None
if tf is None:
    model_error = "TensorFlow could not initialize. Prediction is unavailable."
else:
    try:
        with st.spinner(f"Verifying {selected_disease} model artifacts…"):
            if selected_disease == "Pneumonia":
                selected_model, selected_mapping = load_pneumonia_model(
                    str(PNEUMONIA_MODEL_PATH), str(PNEUMONIA_CONFIG_PATH)
                )
            else:
                selected_model, selected_mapping, selected_preprocessing = load_tuberculosis_model(
                    str(TUBERCULOSIS_MODEL_PATH),
                    str(TUBERCULOSIS_MAPPING_PATH),
                    str(TUBERCULOSIS_PREPROCESSING_PATH),
                )
    except Exception as error:
        model_error = str(error)

with right:
    st.markdown('<div class="section-label">02 / AI READOUT</div>', unsafe_allow_html=True)
    status = "READY" if selected_model is not None and selected_mapping else "UNAVAILABLE"
    st.caption(f"DISEASE / {selected_disease.upper()} &nbsp;&nbsp; • &nbsp;&nbsp; MODEL / {model_label} &nbsp;&nbsp; • &nbsp;&nbsp; STATUS / {status}")
    if model_error:
        st.error(f"{selected_disease} model unavailable: {model_error}")
    elif selected_model is None or not selected_mapping:
        st.error("Model or class mapping could not be verified; prediction is disabled.")
    elif upload_is_valid and st.button("RUN AI CLASSIFICATION", type="primary", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix.casefold()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(upload_bytes)
                temporary_path = Path(temporary_file.name)
            with st.spinner(f"Running the {selected_disease} model and generating Grad-CAM…"):
                if selected_disease == "Pneumonia":
                    result = predict_xray(
                        selected_model,
                        temporary_path,
                        image_size=int(selected_mapping["image_size"]),
                    )
                    result["disease"] = "Pneumonia"
                else:
                    result = predict_tuberculosis(
                        selected_model,
                        temporary_path,
                        selected_mapping,
                        selected_preprocessing,
                    )
                result["upload_digest"] = upload_digest
            st.session_state["result"] = result
        except Exception as error:
            st.session_state.pop("result", None)
            st.error(f"{selected_disease} inference failed: {error}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    result = st.session_state.get("result")
    if result:
        st.caption(f"Selected disease: {result['disease']}")
        st.markdown(f'<p class="readout">{result["predicted_class"]}</p>', unsafe_allow_html=True)
        st.metric("Prediction confidence", f'{result["confidence"]:.1%}')
        st.write("Class probabilities")
        for class_name, probability in result["probabilities"].items():
            st.progress(probability, text=f"{class_name}  {probability:.1%}")
        st.caption(f"Model: ResNet50 transfer learning · Grad-CAM layer: {result['gradcam_layer']}")
    elif selected_model is not None:
        st.info(f"Upload a chest X-ray, then select RUN AI CLASSIFICATION to use the {selected_disease} model.")

st.divider()
st.markdown('<div class="section-label">03 / EXPLAINABILITY</div>', unsafe_allow_html=True)
if st.session_state.get("result"):
    result = st.session_state["result"]
    col1, col2, col3 = st.columns(3)
    col1.image(result["original"], caption="Original (resized input)", use_container_width=True)
    col2.image(result["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
    col3.image(result["overlay"], caption="Heatmap overlay", use_container_width=True)
    st.caption("Grad-CAM is an interpretability visualization of image regions contributing to this model prediction. It does not establish a clinically correct lesion or diagnosis.")
else:
    st.caption("A Grad-CAM heatmap and overlay will appear after a saved trained model processes an uploaded image.")

st.markdown('<p class="disclaimer">This application is an educational/research prototype. It is not a certified medical device, is not intended for clinical diagnosis, and must not replace professional medical judgment.</p>', unsafe_allow_html=True)
