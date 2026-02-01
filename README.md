# 🎭 Face & Gesture Filter (MediaPipe)

A real-time **face and hand gesture filter** built with **MediaPipe** and **OpenCV**, inspired by Instagram/TikTok filters — and a little bit by *Mission: Impossible* 😅.

The project detects facial landmarks and hand gestures from a webcam feed and applies different visual filters dynamically.

---

## ✨ Features

- Real-time face landmark detection (478 points)
- Hand gesture recognition
- Dynamic scaling and positioning of filters
- Multiple filters triggered by gestures
- Runs directly from your webcam

---

## 🧠 Tech Stack

- **Python**
- **MediaPipe**
  - Face Landmarker
  - Gesture Recognizer
- **OpenCV**
- **NumPy**

---

## 🖐 Gesture Mapping

- 🙂 **No gesture (default)** → Glasses + moustache  
- 👍 **Thumbs up** → Crown  
- ✌ **Victory / Peace** → Mask  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
git clone https://github.com/your-username/face-gesture-filter.git
cd face-gesture-filter
### 2️⃣ Install dependencies
pip install -r requirements.txt
### 3️⃣ Download MediaPipe models
Download the official MediaPipe .task models and place them inside the models/ folder:

face_landmarker.task

gesture_recognizer.task

Official MediaPipe documentation and models:
https://developers.google.com/mediapipe

Expected structure:

models/
├── face_landmarker.task
└── gesture_recognizer.task
### 4️⃣ Run the project
python main.py
Press q to quit.

📁 Project Structure
face-gesture-filter/
├── filters/
│   ├── glasses.png
│   ├── moustache.png
│   ├── mask.png
│   └── crown.png
├── models/
│   └── (MediaPipe .task models – not versioned)
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

😅 Notes & Lessons Learned
Not all PNGs from Google are real PNGs.
Some come with fake transparency and huge resolutions (8K+).

MediaPipe makes real-time computer vision surprisingly accessible.

Small geometry mistakes become very visible on a human face.

📌 Inspiration
Inspired by face recognition scenes from the Mission: Impossible movie series and curiosity about real-time biometric systems.