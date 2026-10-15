# Military Aircraft Classification

## Project Overview
Military Aircraft Classification is a project that uses a Convolutional Neural Network (CNN) to classify aircraft images into different categories. It is developed using Python, TensorFlow, and Keras, with a Streamlit web interface for image prediction.

This project demonstrates the application of machine learning and computer vision in the defence domain.

## Objectives
- Classify military aircraft images using a trained CNN model.
- Apply deep learning techniques to image classification.
- Evaluate the trained model using accuracy, a classification report, and a confusion matrix.
- Build a simple web application to make predictions from uploaded images.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Scikit-learn
- Matplotlib
- Pillow
- Streamlit

## Aircraft Categories
The model was trained to recognize these 10 classes:

- B-1
- B-2
- B-52
- C-5
- C-17
- C-130
- C-135
- E-3
- KC-10
- BareLand (background class)

## Model Details
The project uses a Convolutional Neural Network (CNN) for image classification. Images are converted to RGB, resized to 130 × 130 pixels, and normalized before prediction.

The dataset contains 6,123 labeled images. The model was evaluated using a held-out test split and achieved approximately 90% test accuracy on that split.

## Project Structure
```text
Military-Aircraft-Classification/
├── app.py
├── train_model.py
├── predict.py
├── inspect_dataset.py
├── export_sample.py
├── export_samples.py
├── class_names.json
├── aircraft_cnn.keras
├── requirements.txt
├── accuracy_plot.png
├── confusion_matrix.png
└── dataset/
    └── TYPE-NAMES.txt
```

## Dataset Download and Setup

The training dataset is not included in this repository because of its large file size (approximately 229 MB).

**Download the dataset:**

- [Download the dataset from Google Drive](https://drive.google.com/drive/folders/1Wprta1S1azZqM9FRxYjU0mHlHLmoEPFO?usp=sharing)

After downloading the dataset, place `training_dataset.npz` inside the `dataset/` folder so the path is:

```text
dataset/training_dataset.npz
```

The `dataset/` folder should contain:

- `training_dataset.npz`
- `TYPE-NAMES.txt`

Once the dataset is in place, you can train the model by running:

```bash
python train_model.py
```
## Installation

Clone the repository:

```bash
git clone https://github.com/mahak005/Military-Aircraft-Classification
cd Military-Aircraft-Classification
```

Create and activate a virtual environment (optional but recommended).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web Application

```bash
python -m streamlit run app.py
```

Upload an aircraft image and click Classify Image to view the predicted class and confidence.

## Run Prediction from the Terminal

```bash
python predict.py "sample_aircraft.jpg"
```

## Train the Model

After placing the dataset at `dataset/training_dataset.npz`, run:

```bash
python train_model.py
```

## Limitations
- Predictions depend on image quality and the categories represented in the training dataset.
- Test accuracy does not guarantee the same performance on new, real-world images.
- The model is an educational prototype and is not intended for operational defence decisions.

## Future Improvements
- Expand the training dataset with more diverse aircraft images.
- Improve performance on unseen images.
- Explore transfer learning using pretrained CNN architectures.
- Deploy the application online.

## Author
Mahak Soni
