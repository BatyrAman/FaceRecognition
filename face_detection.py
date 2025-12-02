import cv2
import numpy as np
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        x_data = train[i, :-1]      # данные
        y_label = train[i, -1]      # метка класса

        d = distance(test, x_data)  # расстояние
        dist.append([d, y_label])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]

    out = np.unique(labels, return_counts=True)
    index = np.argmax(out[1])
    return out[0][index]


dataset_path = "./face_dataset/"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
labels = []
names = {}
class_id = 0

for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        print("Loading:", fx)

        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("Dataset shape:", trainset.shape)
print("Classes:", names)


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

        out = knn(trainset, face_section.flatten())
        name = names[int(out)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
