
import numpy as np
from PIL import Image

dataset_path = "dataset/training_dataset.npz"
output_path = "sample_aircraft.jpg"

with np.load(dataset_path, allow_pickle=True) as data:
    dataset = data["a"]

# Find the first valid RGB or RGBA image
for row in dataset:
    image = np.asarray(row[0])

    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = image[:, :, :3]
        Image.fromarray(image).save(output_path)
        print("Sample image saved:", output_path)
        print("Original label:", row[1])
        break
else:
    print("No valid image found.")

