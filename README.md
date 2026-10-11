# Emotion Detection from Facial Expressions

This project implements and compares two different approaches for emotion detection from facial expressions:
1. Convolutional Neural Network (CNN)
2. Support Vector Machine (SVM) with HOG features

## Dataset

The project uses the FER-2013 (Facial Expression Recognition 2013) dataset, which contains:
- 48x48 grayscale face images
- 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral

## Project Structure

- `data_preprocessing.py`: Handles data loading and preprocessing
- `cnn_model.py`: Implements the CNN model
- `svm_model.py`: Implements the SVM model with HOG features
- `main.py`: Main script to run the comparison
- `requirements.txt`: Required Python packages

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the FER-2013 dataset from Kaggle and place it in the project directory as `fer2013.csv`

## Usage

Run the main script to train and compare both models:
```bash
python main.py
```

The script will:
1. Load and preprocess the FER-2013 dataset
2. Train both CNN and SVM models
3. Evaluate their performance
4. Generate comparison plots and metrics
5. Save the trained models

## Output

The script generates several output files:
- `training_history.png`: CNN training history plots
- `cnn_confusion_matrix.png`: CNN confusion matrix
- `svm_confusion_matrix.png`: SVM confusion matrix
- `emotion_cnn_model.h5`: Trained CNN model
- `emotion_svm_model.joblib`: Trained SVM model
- `emotion_svm_scaler.joblib`: SVM feature scaler

## Model Comparison

The script provides a detailed comparison of both models, including:
- Accuracy
- Training time
- Classification reports
- Confusion matrices

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- OpenCV
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn 