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