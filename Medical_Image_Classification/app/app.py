"""MEDVISION AI: Streamlit research-prototype interface for a real saved Keras model."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
import typing
try:
    import tensorflow as tf
except ImportError:
    tf = None
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from predict import predict_xray  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "final_model.keras"

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
def load_model(path: str) -> typing.Any:
    if tf is None: return None
    return tf.keras.models.load_model(path)


st.markdown("""
<div class="hero"><h1>MEDVISION AI</h1><p>DEEP LEARNING-BASED CHEST X-RAY IMAGE CLASSIFICATION · RESEARCH CONSOLE</p></div>
""", unsafe_allow_html=True)

model_available = MODEL_PATH.exists()
status = "READY" if model_available else "MODEL ARTIFACT REQUIRED"
st.caption(f"MODEL / ResNet50 &nbsp;&nbsp; • &nbsp;&nbsp; STATUS / {status}")

left, right = st.columns((1, 1.25), gap="large")
with left:
    st.markdown('<div class="section-label">01 / IMAGE INTAKE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a compatible chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        preview = Image.open(uploaded_file).convert("RGB")
        st.image(preview, caption="Uploaded image preview", use_container_width=True)

with right:
    st.markdown('<div class="section-label">02 / AI READOUT</div>', unsafe_allow_html=True)
    if not model_available:
        st.warning("A trained final ResNet50 model has not yet been placed in `models/resnet50/final_model.keras`.")
    elif uploaded_file and st.button("RUN AI CLASSIFICATION", type="primary", use_container_width=True):
        with st.spinner("Running the saved model and generating Grad-CAM…"):
            suffix = Path(uploaded_file.name).suffix or ".jpeg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(uploaded_file.getvalue())
                temporary_path = Path(temporary_file.name)
            try:
                result = predict_xray(load_model(str(MODEL_PATH)), temporary_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result:
        st.markdown(f'<p class="readout">{result["predicted_class"]}</p>', unsafe_allow_html=True)
        st.metric("Prediction confidence", f'{result["confidence"]:.1%}')
        st.write("Class probabilities")
        st.progress(result["probabilities"]["NORMAL"], text=f'NORMAL  {result["probabilities"]["NORMAL"]:.1%}')
        st.progress(result["probabilities"]["PNEUMONIA"], text=f'PNEUMONIA  {result["probabilities"]["PNEUMONIA"]:.1%}')
        st.caption(f"Model: ResNet50 transfer learning · Grad-CAM layer: {result['gradcam_layer']}")
    elif model_available:
        st.info("Upload an image, then select RUN AI CLASSIFICATION. All displayed values will come from the saved model.")

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
