import os
import json
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image

MODEL_PATH = "aircraft_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = (130, 130)


def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def predict_image(image_path, model, class_names):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    print("\n--- Aircraft Classification Result ---")
    print(f"Image: {image_path}")
    print(f"Predicted category: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop 3 predictions:")
    top_indices = np.argsort(probabilities)[::-1][:3]

    for index in top_indices:
        print(
            f"{class_names[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a military aircraft image using CNN."
    )

    parser.add_argument(
        "image",
        help="Path to the image you want to classify"
    )

    args = parser.parse_args()

    model, class_names = load_model_and_classes()
    predict_image(args.image, model, class_names)
