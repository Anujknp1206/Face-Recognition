import cv2
import numpy as np
from PIL import Image
import os

# Define dataset path
dataset_path = 'dataset'
trainer_path = 'trainer.yml'

# Initialize recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get images and labels
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    unique_ids = {}

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        file_name = os.path.split(imagePath)[-1]
        parts = file_name.split(".")
        if len(parts) < 3:
            continue  # Skip incorrectly named files

        user_id = parts[1]

        # Convert text IDs to numeric IDs
        if user_id not in unique_ids:
            unique_ids[user_id] = len(unique_ids) + 1  # Assign a unique numeric ID

        numeric_id = unique_ids[user_id]

        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(numeric_id)

    return face_samples, ids

print("[INFO] Training faces. It may take a few seconds...")
faces, ids = getImagesAndLabels(dataset_path)

if faces:
    recognizer.train(faces, np.array(ids))
    recognizer.write(trainer_path)
    print(f"\n[INFO] {len(set(ids))} faces trained successfully.")
else:
    print("\n[ERROR] No faces found for training.")
