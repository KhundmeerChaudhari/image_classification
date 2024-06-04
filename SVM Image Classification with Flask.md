# SVM Image Classification with Flask

This project demonstrates image classification using Support Vector Machines (SVM) with Flask for building a web application.

## Overview

The project consists of three main parts:

1. **Feature Extraction**: This part involves extracting features from images to represent them as numerical data suitable for machine learning. Features are extracted using histogram equalization, image blurring, and resizing.

2. **Training SVM Model**: The preprocessed features are used to train an SVM model using scikit-learn.

3. **Flask Web Application**: The trained SVM model is loaded into a Flask web application to classify images uploaded by users.

## Feature Extraction

- Images are preprocessed using the following steps:
  - **Histogram Equalization**: Enhances the contrast of images by spreading out the intensity values.
  - **Image Blurring**: Reduces noise and details in images using a Gaussian blur filter.
  - **Resizing**: Standardizes the size of images to a common size (e.g., 100x100 pixels).

## Training SVM Model

### Dataset Preparation

- The dataset is provided in the `dataset_full/` directory, containing images of various categories.
- Features are extracted from images and saved as numerical data in a CSV file (`classify2.csv`).

### Model Training

- The `train_svm.py` script preprocesses the dataset, splits it into training and testing sets, and trains an SVM model.
- The SVM model is trained using the Radial Basis Function (RBF) kernel with hyperparameter C=0.5 and degree=1.
- The trained model is saved as `model.sav` using joblib.

## Flask Web Application

- The Flask web application (`app.py`) provides a user interface for image classification.
- Users can upload images through a web form.
- Upon image upload, the uploaded image is preprocessed, and features are extracted.
- The extracted features are used to classify the image using the trained SVM model.
- The predicted class for each image is displayed on the result page.

### Project Structure

- `dataset_full/`: Directory containing the full dataset.
- `classify2.csv`: CSV file containing preprocessed features used for training.
- `model.sav`: Trained SVM model saved using joblib.
- `uploads/`: Directory where uploaded images are stored temporarily.
- `templates/`: Directory containing HTML templates for the web application.
- `train_svm.py`: Python script for training the SVM model.
- `app.py`: Python script for the Flask web application.
- `README.md`: This file.

### Requirements

- Python 3
- Flask
- NumPy
- OpenCV (cv2)
- scikit-learn


