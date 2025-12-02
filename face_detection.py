import cv2
import numpy as np
import os


dataset_path = "./face_dataset/"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 10

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(frame.shape[1], x + w + offset)
        y2 = min(frame.shape[0], y + h + offset)

        face_section = frame[y1:y2, x1:x2]

        if face_section.size == 0:
            continue

        face_section = cv2.resize(face_section, (200, 200))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
