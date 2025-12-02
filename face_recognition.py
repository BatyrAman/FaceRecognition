import cv2
import numpy as np
import os
from sklearn.decomposition import PCA

# -------------------- BASIC FUNCTIONS --------------------

def normalize(vec):
    mean = np.mean(vec)
    std = np.std(vec) + 1e-10
    return (vec - mean) / std

def pca_fit(train_data, n_components=100):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(train_data)
    return pca, transformed

def advanced_knn_pca(train, test, pca, k=5, threshold=150.0):
    test = normalize(test)
    test_pca = pca.transform([test])[0]  # convert test sample to PCA space

    dist_list = []

    for i in range(train.shape[0]):
        x_data = train[i, :-1]   # PCA data
        y_label = train[i, -1]

        d = np.linalg.norm(test_pca - x_data)
        dist_list.append((d, y_label))

    dist_list = sorted(dist_list, key=lambda x: x[0])[:k]

    # Weighted voting
    weights = {}
    for d, label in dist_list:
        w = 1 / (d + 1e-6)
        weights[label] = weights.get(label, 0) + w

    best = max(weights, key=weights.get)


    return best


# -------------------- LOAD DATASET --------------------

dataset_path = "./face_dataset/"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
labels = []
names = {}
class_id = 0

for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]

        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)

        class_id += 1

# Create training arrays
X = np.concatenate(face_data, axis=0)   # All images
y = np.concatenate(labels, axis=0)      # Labels

print("Loaded dataset:", X.shape)
print("Labels:", y.shape)

# -------------------- TRAIN PCA --------------------

pca, X_pca = pca_fit(X, n_components = min(30, X.shape[0]))

# Combine PCA data with labels:
train_pca = np.hstack([X_pca, y.reshape(-1, 1)])
print("PCA trainset:", train_pca.shape)

# -------------------- CAMERA LOOP --------------------
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
        test_vec = face_section.flatten()

        out = advanced_knn_pca(train_pca, test_vec, pca)



        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, names[int(out)], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Face Recognition PCA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
