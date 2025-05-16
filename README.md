# 🧠 AI-Powered Attendance System 🎓

This is an AI-powered attendance system that captures classroom images, detects student faces, and matches them against a pre-registered database to mark attendance automatically. It's built using **Flask**, **OpenCV**, and **Face Recognition** libraries.

## 🚀 Features

- 🔐 Secure login for admin access
- 📸 Capture real-time class image using webcam
- 📁 Upload classroom images from local system
- 🤖 Automatic face detection from captured/uploaded images
- 🧠 Face matching using biometric features (face embeddings)
- ✅ Attendance generation and display via web dashboard
- 💾 Images are stored locally in a structured directory

## 🛠️ Tech Stack

- **Backend:** Python (Flask)
- **Computer Vision:** OpenCV
- **Face Recognition:** `face_recognition` library
- **Frontend:** HTML5, CSS (via Flask templates)
- **Storage:** Local file system

## 📂 Project Structure
AI-Powered-Attendance-System/
├── app.py # Main Flask app
├── capture_class_images.py # Webcam image capture logic
├── face_detection.py # Face detection from images
├── face_matching.py # Face matching with saved data
├── templates/ # HTML templates (login, dashboard, attendance)
└── class_images/ # Saved classroom images

## 🧪 How It Works

1. **Login** as an admin
2. **Capture** or **upload** a classroom photo
3. **Detect** all visible faces in the image
4. **Match** detected faces with stored student data
5. **Display** attendance list on dashboard
