# 1. Import libraries
import cv2
import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical, load_img, img_to_array

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Name:", tf.test.gpu_device_name())

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# 2. Download dataset using kagglehub
from pathlib import Path
import shutil

# Create a data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download dataset from Kaggle
download_path = Path(kagglehub.dataset_download("valentynsichkar/traffic-signs-preprocessed"))
print(f"Download path: {download_path}")

# Copy all files to our data directory
for file_path in download_path.rglob("*.*"):
    if file_path.is_file():
        shutil.copy2(file_path, data_dir)
        print(f"Copied: {file_path.name}")

# 3. Load the dataset
# The dataset provides train.p, valid.p, test.p as pickled files
import pickle

train_path = data_dir / "train.pickle"
valid_path = data_dir / "valid.pickle"
test_path = data_dir / "test.pickle"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)
with open(valid_path, "rb") as f:
    valid_data = pickle.load(f)
with open(test_path, "rb") as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Add image preprocessing with OpenCV
def preprocess_image(image):
    # Convert to BGR (OpenCV format)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Apply Gaussian blur to reduce noise
    image_cv = cv2.GaussianBlur(image_cv, (3, 3), 0)
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    # Convert back to RGB
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return enhanced

# Apply preprocessing to all images
X_train = np.array([preprocess_image(img) for img in X_train])
X_val = np.array([preprocess_image(img) for img in X_val])
X_test = np.array([preprocess_image(img) for img in X_test])

# 4. Normalize images
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# 5. One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# 6. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 7. Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=64)

# 8. Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

# 9. Confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12,8))
sns.heatmap(cm, cmap="Blues", annot=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred_classes))
# 10. Save the model
model_path = "traffic_sign_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# 11. Load the model (for inference or further training)
loaded_model = models.load_model(model_path)
print("Model loaded.")

# 12. Predict function for new images
def detect_and_crop_sign(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for red signs
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask1 + mask2
    
    # Create mask for blue signs
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_red, mask_blue)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw rectangle around the sign
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the sign
        sign = original[y:y+h, x:x+w]
        
        # Resize to 32x32
        sign = cv2.resize(sign, (32, 32))
        
        return image, sign, (x, y, w, h)
    
    return image, None, None

def predict_sign(image_path, model):
    # Detect and crop the sign
    image, cropped_sign, bbox = detect_and_crop_sign(image_path)
    
    if cropped_sign is None:
        return None, image, None
    
    # Preprocess the cropped sign
    img_array = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2RGB)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, image, confidence


