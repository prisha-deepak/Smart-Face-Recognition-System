# 🎭 Face Recognition Attendance System  

## 📌 Overview  
This project is a **Face Recognition-based Attendance System** that detects faces from a live webcam feed and marks attendance based on recognized individuals. It utilizes **OpenCV’s LBPH (Local Binary Patterns Histograms) Face Recognizer** to identify known faces and logs attendance in real-time.  

## 🚀 Features  
- 🎭 **Real-time Face Recognition** using OpenCV  
- 📂 **Pre-trained Model** for recognizing known faces  
- 🕵️ **Duplicate Detection Prevention** – Only logs a name once per session  
- ⏱️ **Timestamp Logging** – Records the time of recognition  
- 🎨 **Live Display with Bounding Boxes & Labels**  

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries:** OpenCV, NumPy, OS  
- **Model Used:** LBPH Face Recognizer  
- **Hardware:** Webcam (for real-time face detection)  

## 📂 Folder Structure  
```
📂 Face-Recognition-Attendance  
│── 📁 known_faces  # Directory for storing images of known faces  
│── 📜 main.py  # Main script for real-time face recognition  
│── 📜 requirements.txt  # Dependencies list  
```


## 📖 How It Works  
1. **Load Known Faces** – The system scans the `known_faces/` directory and assigns a unique label to each person.  
2. **Face Detection & Recognition** –  
   - Captures frames from the webcam.  
   - Detects faces using OpenCV’s **Haarcascade Classifier**.  
   - Compares detected faces with stored known faces using **LBPH Face Recognizer**.  
3. **Mark Attendance** –  
   - If a known face is recognized, their name and timestamp are recorded.  
   - Prevents duplicate attendance within the same session.  
4. **Live Video Feed** – Displays recognized faces with red bounding boxes and labels.  

