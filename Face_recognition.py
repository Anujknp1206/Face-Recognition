import cv2
import numpy as np
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')  # Ensure the correct path

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# Predefined user names (should match dataset IDs)
names = ["Unknown", "User1", "User2", "User3", "User4", "User5"]

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            name = names[id] if id < len(names) else "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Exit on 'ESC' key
        break

print("[INFO] Exiting program")
cam.release()
cv2.destroyAllWindows()
