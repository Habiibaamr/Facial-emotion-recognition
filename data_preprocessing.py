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