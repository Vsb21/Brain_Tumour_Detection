import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# Project configuration
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# The app checks these locations in order.
MODEL_CANDIDATES = [
    "models/efficientnetb4_best.keras",
    "models/efficientnetb4.keras",
    "efficientnetb4_best.keras",
    "efficientnetb4.keras",
]

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload an MRI image to see the prediction made by the trained "
    "EfficientNetB4 model."
)
st.caption("Academic project demonstration only. This tool does not provide a medical diagnosis.")

def find_model_path():
    for candidate in MODEL_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None

@st.cache_resource
def load_model(model_path):
    # compile=False avoids loading training-only optimizer/loss state.
    return tf.keras.models.load_model(model_path, compile=False)

model_path = find_model_path()

if model_path is None:
    st.error(
        "EfficientNetB4 model file was not found. Add "
        "`efficientnetb4_best.keras` or `efficientnetb4.keras` to the "
        "repository root or to the `models/` folder."
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.success(f"Model loaded: `{model_path}`")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    st.image(image, caption="Uploaded MRI image", use_container_width=True)

    if st.button("Classify MRI", type="primary", use_container_width=True):
        try:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            height = int(input_shape[1] or 224)
            width = int(input_shape[2] or 224)

            resized = image.resize((width, height))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            batch = np.expand_dims(array, axis=0)

            with st.spinner("Analyzing image..."):
                probabilities = model.predict(batch, verbose=0)[0]

            if len(probabilities) != len(CLASSES):
                st.error(
                    "The model output does not match the four configured "
                    "classes. Check the class order used during training."
                )
                st.stop()

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASSES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100.0

            st.subheader("Prediction")
            st.metric("Predicted class", predicted_class.replace("_", " ").title())
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Class probabilities")
            for class_name, probability in zip(CLASSES, probabilities):
                percentage = float(probability) * 100.0
                st.write(f"**{class_name.title()}**: {percentage:.2f}%")
                st.progress(min(max(float(probability), 0.0), 1.0))

            st.warning(
                "This result is an experimental model prediction, not a "
                "medical diagnosis. Please consult a qualified radiologist "
                "or healthcare professional for medical interpretation."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Model: EfficientNetB4 | Input: resized to the model's saved input dimensions")
