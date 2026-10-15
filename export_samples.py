import os
import numpy as np
from PIL import Image

DATASET_PATH = "dataset/training_dataset.npz"
OUTPUT_DIR = "sample_images"
IMAGES_PER_CLASS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

with np.load(DATASET_PATH, allow_pickle=True) as data:
    dataset = data["a"]

class_counts = {}

for row in dataset:
    image = np.asarray(row[0])
    label = int(row[1])

    # Keep the first 5 images from each category
    if class_counts.get(label, 0) >= IMAGES_PER_CLASS:
        continue

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        continue

    # Convert RGB/RGBA to RGB
    image = image[:, :, :3].astype(np.uint8)

    class_folder = os.path.join(OUTPUT_DIR, f"class_{label}")
    os.makedirs(class_folder, exist_ok=True)

    count = class_counts.get(label, 0) + 1
    image_path = os.path.join(
        class_folder, f"sample_{count}.jpg"
    )

    Image.fromarray(image).save(image_path)
    class_counts[label] = count

print("\nExtraction completed!")

total = 0
for label in sorted(class_counts):
    count = class_counts[label]
    total += count
    print(f"Class {label}: {count} images")

print(f"\nTotal images exported: {total}")
print(f"Saved in: {OUTPUT_DIR}")
