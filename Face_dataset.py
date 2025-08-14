import cv2
import os

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Load face detector
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Get user ID
face_id = input('\n Enter user ID and press Enter: ')

print('Initializing face capture...')

# Ensure dataset folder exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        count += 1

        cv2.imwrite(f"{dataset_path}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow("Image", img)

    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 30:  # Exit if 'ESC' is pressed or 30 samples are captured
        break

print("\n[INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
