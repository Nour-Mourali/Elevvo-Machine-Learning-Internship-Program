import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import pandas as pd
import os

def test_image(image_path, model, labels=None):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100

    # Display results
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_img)
    
    if labels is not None:
        title = f'Predicted: {labels[predicted_class]}\nConfidence: {confidence:.2f}%'
    else:
        title = f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%'
    
    plt.title(title)
    plt.axis('off')
    plt.show()

    # Print prediction details
    print("\nPrediction Details:")
    print(f"Class Number: {predicted_class}")
    if labels is not None:
        print(f"Sign Name: {labels[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # Load model
    model_path = "traffic_sign_model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        exit()

    model = models.load_model(model_path)
    print("Model loaded successfully!")

    # Load labels if available
    try:
        labels_df = pd.read_csv('data/label_names.csv')
        labels = labels_df['SignName'].values
    except:
        print("Warning: Could not load sign names")
        labels = None

    # Test image
    test_path = "testt.jpg"
    if not os.path.exists(test_path):
        print(f"Error: Test image {test_path} not found")
        exit()

    # Run test
    test_image(test_path, model, labels)