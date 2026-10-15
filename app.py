import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# 1. Page configuration
# -----------------------------

st.set_page_config(
    page_title="Military Aircraft Classification",
    page_icon="✈️",
    layout="centered"
)

MODEL_PATH = "aircraft_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (130, 130)

# -----------------------------
# 2. Load trained model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# -----------------------------
# 3. App interface
# -----------------------------

st.title("✈️ Military Aircraft Classification")

st.write(
    "Upload an aircraft image to classify it using a "
    "Convolutional Neural Network (CNN)."
)

st.info(
    "The model recognizes nine aircraft categories and one "
    "background category: BareLand."
)

# Check required files
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please check aircraft_cnn.keras.")
    st.stop()

if not os.path.exists(CLASS_NAMES_PATH):
    st.error("Class names file not found. Please check class_names.json.")
    st.stop()

# Load model and categories
try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# -----------------------------
# 4. Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, caption="Image selected for classification",
                 use_container_width=True)

        if st.button("Classify Image", type="primary",
                     use_container_width=True):

            with st.spinner("Classifying image..."):
                resized_image = image.resize(IMAGE_SIZE)

                image_array = (
                    np.asarray(resized_image, dtype=np.float32) / 255.0
                )

                image_array = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))

                predicted_class = class_names[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100

            st.success("Classification completed!")

            st.subheader("Prediction Result")
            st.markdown(f"### Predicted Category: **{predicted_class}**")

            st.metric("Model Confidence", f"{confidence:.2f}%")

            # Show the three most likely categories
            st.subheader("Top Predictions")

            top_indices = np.argsort(probabilities)[::-1][:3]

            for index in top_indices:
                score = float(probabilities[index]) * 100
                st.write(f"**{class_names[index]}** — {score:.2f}%")
                st.progress(float(probabilities[index]))

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")

# -----------------------------
# 5. Project information
# -----------------------------

with st.expander("About this project"):
    st.write(
        "This project uses a Convolutional Neural Network (CNN) "
        "to classify images into ten categories."
    )

    st.write("**Model:** CNN")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input image size:** 130 × 130 pixels")
    st.write("**Number of categories:** 10")
    st.write("**Test accuracy:** 90% on the current test set")

    st.write(
        "**Categories:** " + ", ".join(class_names)
    )

st.divider()
st.caption("Machine Learning Lab Mini-Project | CNN Image Classification")
