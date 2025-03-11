import cv2
import numpy as np
import os
from datetime import datetime

# Path to the directory containing images of known faces
known_faces_dir = 'C:\developments\\attendance\known_faces'

# Initialize the face recognizer and the face detector
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces and their labels
faces = []
labels = []
label_map = {}

current_label = 0

# Load each known face and assign a label
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces_detected = face_cascade.detectMultiScale(image)
        for (x, y, w, h) in faces_detected:
            face = image[y:y + h, x:x + w]
            faces.append(face)
            labels.append(current_label)
        label_map[current_label] = os.path.splitext(filename)[0]
        current_label += 1

# Train the face recognizer
face_recognizer.train(faces, np.array(labels))
# Initialize some variables
face_locations = []
face_names = []

# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)
from datetime import datetime

from datetime import datetime

# Set to keep track of names that have already been printed
printed_names = set()

def mark_attendance(name):
    global printed_names
    
    # Check if the name has already been printed
    if name not in printed_names:
        # Generate current date/time string without seconds
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Print the name and date/time
        print(f'Name: {name}, Time: {current_time}')
        
        # Add the name to the set of printed names
        printed_names.add(name)



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    face_names = []
    for (x, y, w, h) in faces_detected:
        face = gray_frame[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face)
        name = label_map.get(label, "Unknown")
        face_names.append((x, y, w, h, name))
        mark_attendance(name)  

    # Display the results
    for (x, y, w, h, name) in face_names:
        # Draw a box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
