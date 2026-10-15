import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1. Configuration
# -----------------------------

DATASET_PATH = os.path.join("dataset", "training_dataset.npz")
CLASS_NAMES_PATH = os.path.join("dataset", "TYPE-NAMES.txt")

MODEL_PATH = "aircraft_cnn.keras"
CLASS_NAMES_OUTPUT = "class_names.json"

IMAGE_SIZE = 130
TEST_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 20

# -----------------------------
# 2. Load class names
# -----------------------------

print("Loading aircraft class names...")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = file.read().replace('"', "").split()

print("Classes:", class_names)
print("Number of classes:", len(class_names))

# -----------------------------
# 3. Load dataset
# -----------------------------

print("\nLoading dataset...")

# allow_pickle=True is required because this dataset stores
# image arrays and labels inside an object array.
with np.load(DATASET_PATH, allow_pickle=True) as data:
    dataset = data["a"]


# Convert every image to RGB by keeping only the first 3 channels.
images = np.stack([
    np.asarray(row[0])[:, :, :3]
    for row in dataset
])

labels = np.array(
    [int(row[1]) for row in dataset],
    dtype=np.int32
)


print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Total images:", len(images))

# -----------------------------
# 4. Validate dataset
# -----------------------------

if len(class_names) != len(np.unique(labels)):
    raise ValueError(
        "The number of class names does not match the unique labels."
    )

if labels.min() < 0 or labels.max() >= len(class_names):
    raise ValueError("Dataset contains an invalid label.")

if images.ndim != 4 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
    raise ValueError("Expected RGB images of size 130 x 130.")

# -----------------------------
# 5. Prepare images
# -----------------------------

print("\nPreparing images...")

images = images.astype(np.float32) / 255.0

# -----------------------------
# 6. Split training and testing data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# -----------------------------
# 7. Build CNN model
# -----------------------------

print("\nBuilding CNN model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 8. Train the model
# -----------------------------

print("\nStarting CNN training...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------
# 9. Evaluate the model
# -----------------------------

print("\nEvaluating model on test data...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# -----------------------------
# 10. Generate predictions
# -----------------------------

print("\nGenerating predictions...")

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

# -----------------------------
# 11. Save confusion matrix
# -----------------------------

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(12, 10))
display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Military Aircraft Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Saved confusion_matrix.png")

# -----------------------------
# 12. Save accuracy graph
# -----------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=150)
plt.close()

print("Saved accuracy_plot.png")

# -----------------------------
# 13. Save trained model
# -----------------------------

print("\nSaving trained model...")

model.save(MODEL_PATH)

with open(CLASS_NAMES_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("Saved model:", MODEL_PATH)
print("Saved class names:", CLASS_NAMES_OUTPUT)
print("\nTraining completed successfully!")

