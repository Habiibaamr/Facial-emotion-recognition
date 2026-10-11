# Project Code Overview

This file aggregates the current source files so you can view and discuss the full code in one place.

---

## `cnn_model.py`

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class EmotionCNN:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=50):
        callbacks = [
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
```

---

## `svm_model.py`

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
import joblib

class EmotionSVM:
    def __init__(self):
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        
    def extract_hog_features(self, images):
        """Extract HOG features from images"""
        hog = cv2.HOGDescriptor(
            (48, 48),  # winSize
            (16, 16),  # blockSize
            (8, 8),    # blockStride
            (8, 8),    # cellSize
            9          # nbins
        )
        
        features = []
        for img in images:
            # If image has channel dim (e.g., (H,W,1)), squeeze it
            if img.ndim == 3 and img.shape[2] == 1:
                img2d = img[:, :, 0]
            elif img.ndim == 3 and img.shape[2] != 1:
                # If there are multiple channels, convert to grayscale
                img2d = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                img2d = img

            # If image is float in [0,1], convert to uint8 [0,255]
            if np.issubdtype(img2d.dtype, np.floating):
                img_uint8 = (img2d * 255.0).astype(np.uint8)
            else:
                img_uint8 = img2d.astype(np.uint8)

            # Ensure image size matches HOG winSize
            win_w, win_h = hog.winSize
            if (img_uint8.shape[1], img_uint8.shape[0]) != (win_w, win_h):
                img_uint8 = cv2.resize(img_uint8, (win_w, win_h))

            # Extract HOG features
            hog_features = hog.compute(img_uint8)
            features.append(hog_features.flatten())
            
        return np.array(features)
    
    def train(self, X_train, y_train):
        """Train the SVM model"""
        # Extract HOG features
        X_train_hog = self.extract_hog_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_hog)
        
        # Train SVM
        self.svm.fit(X_train_scaled, y_train)
        
    def predict(self, X):
        """Make predictions"""
        X_hog = self.extract_hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability estimates"""
        X_hog = self.extract_hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.svm.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, model_path, scaler_path):
        """Save the model and scaler"""
        joblib.dump(self.svm, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """Load the model and scaler"""
        self.svm = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
```

---

## `data_preprocessing.py`

```python
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, img_size=(48, 48)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_fer2013(self, data_path):
        """Load and preprocess FER-2013 dataset"""
        # Check that the CSV exists and provide a helpful error if not
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file not found: '{data_path}'.\n"
                "Please download the FER-2013 CSV and place it in the project folder, "
                "or pass the correct path to `prepare_data()`. See https://www.kaggle.com/datasets/msambare/fer2013 for the dataset."
            )

        data = pd.read_csv(data_path)
        pixels = data['pixels'].tolist()
        width, height = self.img_size
        faces = []
        
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.img_size)
            faces.append(face.astype('float32'))
            
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        
        emotions = self.label_encoder.fit_transform(data['emotion'])
        
        return faces, emotions
    
    def normalize_data(self, X):
        """Normalize pixel values to [0, 1]"""
        return X / 255.0
    
    def augment_data(self, X, y):
        """Apply data augmentation techniques"""
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Work with images that may be normalized floats [0,1] with a channel
            img = X[i]
            # Remove channel dimension for OpenCV ops if present
            if img.ndim == 3 and img.shape[2] == 1:
                img2d = img[:, :, 0]
            else:
                img2d = img

            # Convert to uint8 (0-255) for OpenCV
            img_uint8 = (img2d * 255.0).astype(np.uint8)

            # Original image (keep as original normalized float)
            augmented_X.append(img.astype('float32'))
            augmented_y.append(y[i])

            # Horizontal flip
            flipped_uint8 = cv2.flip(img_uint8, 1)
            flipped = flipped_uint8.astype('float32') / 255.0
            if img.ndim == 3 and img.shape[2] == 1:
                flipped = np.expand_dims(flipped, -1)
            augmented_X.append(flipped)
            augmented_y.append(y[i])

            # Slight rotation
            angle = np.random.uniform(-10, 10)
            rotated_uint8 = self._rotate_image(img_uint8, angle)
            rotated = rotated_uint8.astype('float32') / 255.0
            if img.ndim == 3 and img.shape[2] == 1:
                rotated = np.expand_dims(rotated, -1)
            augmented_X.append(rotated)
            augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def _rotate_image(self, image, angle):
        """Helper function to rotate image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
    def prepare_data(self, data_path, test_size=0.2, augment=True):
        """Prepare data for training"""
        X, y = self.load_fer2013(data_path)
        X = self.normalize_data(X)
        
        if augment:
            X, y = self.augment_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
```

---

## `main.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_preprocessing import DataPreprocessor
from cnn_model import EmotionCNN
from svm_model import EmotionSVM
import time
import os
import sys

def plot_training_history(history):
    """Plot training history for CNN"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and prepare data
    dataset_path = 'fer2013.csv'
    print("Loading and preprocessing data...")
    if not os.path.exists(dataset_path):
        print(\
            "ERROR: dataset file 'fer2013.csv' not found in project directory.\n"
            "Download the FER-2013 CSV (for example from Kaggle) and place it here, "
            "or pass the correct path to DataPreprocessor.prepare_data().\n"
        )
        sys.exit(1)

    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    
    # Train and evaluate CNN
    print("\nTraining CNN model...")
    cnn = EmotionCNN()
    start_time = time.time()
    history = cnn.train(X_train, y_train, X_test, y_test)
    cnn_time = time.time() - start_time
    
    # Plot CNN training history
    plot_training_history(history)
    
    # Evaluate CNN
    cnn_loss, cnn_accuracy = cnn.evaluate(X_test, y_test)
    print(f"\nCNN Results:")
    print(f"Training time: {cnn_time:.2f} seconds")
    print(f"Test accuracy: {cnn_accuracy:.4f}")
    print(f"Test loss: {cnn_loss:.4f}")
    
    # Train and evaluate SVM
    print("\nTraining SVM model...")
    svm = EmotionSVM()
    start_time = time.time()
    svm.train(X_train, y_train)
    svm_time = time.time() - start_time
    
    # Evaluate SVM
    svm_results = svm.evaluate(X_test, y_test)
    print(f"\nSVM Results:")
    print(f"Training time: {svm_time:.2f} seconds")
    print(f"Test accuracy: {svm_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(svm_results['classification_report'])
    
    # Plot confusion matrices
    cnn_predictions = np.argmax(cnn.predict(X_test), axis=1)
    cnn_cm = confusion_matrix(y_test, cnn_predictions)
    plot_confusion_matrix(cnn_cm, "CNN Confusion Matrix")
    plot_confusion_matrix(svm_results['confusion_matrix'], "SVM Confusion Matrix")
    
    # Save models
    cnn.save_model('emotion_cnn_model.h5')
    svm.save_model('emotion_svm_model.joblib', 'emotion_svm_scaler.joblib')
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print(f"{'Model':<10} {'Accuracy':<10} {'Training Time (s)':<20}")
    print("-" * 40)
    print(f"{'CNN':<10} {cnn_accuracy:.4f} {cnn_time:.2f}")
    print(f"{'SVM':<10} {svm_results['accuracy']:.4f} {svm_time:.2f}")

if __name__ == "__main__":
    main()
```

---

## `requirements.txt`

```
numpy>=1.19.2
pandas>=1.2.4
opencv-python>=4.5.1
tensorflow>=2.4.1
scikit-learn>=0.24.2
matplotlib>=3.3.4
imutils>=0.5.4 
h5py>=3.1.0
```

---

## `README.md`

See the project's README for setup and usage instructions. (Already in repository.)

---

If you want, I can:
- split the code into a `src/` package and add a simple `run.py` runner,
- add unit tests or a small sample dataset loader to test locally,
- or generate an annotated summary for each file explaining responsibilities and potential improvements.

Tell me which of those you'd like next.
