
import numpy as np
from collections import Counter

dataset_path = "dataset/training_dataset.npz"

with np.load(dataset_path, allow_pickle=True) as data:
    dataset = data["a"]

print("Total records:", len(dataset))

shapes = Counter()
invalid_examples = []

for index, row in enumerate(dataset):
    try:
        image = np.asarray(row[0])

        if image.ndim == 3 and image.shape[2] == 3:
            shapes[str(image.shape)] += 1
        else:
            shapes[f"Invalid: shape={image.shape}, dtype={image.dtype}"] += 1

            if len(invalid_examples) < 15:
                invalid_examples.append(
                    (index, image.shape, image.dtype, repr(row[0])[:150])
                )

    except Exception as error:
        shapes[f"Conversion error: {type(error).__name__}"] += 1

        if len(invalid_examples) < 15:
            invalid_examples.append(
                (index, "conversion failed", type(error).__name__, str(error))
            )

print("\nRecord types and shapes:")
for shape, count in shapes.items():
    print(f"{shape}: {count}")

print("\nFirst invalid examples:")
for example in invalid_examples:
    print(example)
