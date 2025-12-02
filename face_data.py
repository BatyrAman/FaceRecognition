import cv2
import numpy as np
import os

dataset_path = r"face_dataset/"
os.makedirs(dataset_path, exist_ok=True)

file_name = input("Enter name: ").strip().replace(" ", "_")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
skip = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    print("faces detected:", len(faces))   # DEBUG

    if len(faces) == 0:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    offset = 10
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(frame.shape[1], x + w + offset)
    y2 = min(frame.shape[0], y + h + offset)

    face_offset = frame[y1:y2, x1:x2]

    print("face_offset shape:", face_offset.shape)  # DEBUG

    if face_offset.size == 0:
        continue

    face_selection = cv2.resize(face_offset, (200, 200))

    skip += 1
    if skip % 10 == 0:
        face_data.append(face_selection)
        print("Saved samples:", len(face_data))

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

print("Collected samples:", len(face_data))  # DEBUG

if len(face_data) > 0:
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))

    save_path = os.path.join(dataset_path, file_name + ".npy")
    np.save(save_path, face_data)

    print("Saved to:", save_path)
else:
    print("⚠ ERROR: No samples collected. Nothing saved.")

cap.release()
cv2.destroyAllWindows()
