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