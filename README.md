
# Face Recognition Attendance System 👤📊

A Python-based application that automates attendance tracking using facial recognition technology. This system captures employee/student faces, trains a recognition model, and records attendance automatically when authorized faces are detected.

## Features ✨

- **Face Detection**: Real-time face detection using Haar Cascades
- **Face Recognition**: LBPH (Local Binary Patterns Histograms) algorithm
- **Attendance Tracking**: Automatic recording with timestamps
- **User Management**: Add multiple users with unique IDs
- **Data Export**: Save attendance records to CSV
- **Modern UI**: Intuitive graphical interface with Tkinter

## Prerequisites 🛠️

- Python 3.6+
- Webcam
- Recommended: Good lighting conditions


## How to Use This README

1. Save this content as `README.md` in your project root folder
2. Customize sections (especially URLs and future enhancements) as needed
3. Add a LICENSE file if you want to specify terms
4. Update the requirements.txt with your actual dependencies:

This README provides:
- Clear installation instructions
- Visual project structure
- Step-by-step usage guide
- Troubleshooting section
- Future improvement ideas

  
## Installation Guide 📥

Usage Flow (3-Step Process)
1️⃣ First: Capture Face Dataset
Run to register new users and capture face samples:


python face_dataset.py
Enter User ID when prompted

Face the camera - system will capture 30 samples

Creates images in dataset/ folder (User.{ID}.{Count}.jpg)

2️⃣ Second: Train the Model
Run to process captured faces and train recognizer:


python training.py
Processes all images in dataset/

Generates trainer.yml (trained model file)

Wait for "X faces trained" confirmation

3️⃣ Third: Run Recognition System
Launch the attendance tracking interface:

python Face_Recognition.py
Detects faces in real-time

Automatically records recognized users

Displays confidence percentage

Press ESC to exit 

### 1. Clone the repository
```bash
git clone https://github.com/Anujknp1206/Face-Recognition.git
cd Face-Recognition

