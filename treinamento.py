import cv2 as cv
import os
import numpy as np

# Get the list of people (directories) in the 'fotos' folder
people = os.listdir("./fotos")

# Get the directory of the current script
DIR = os.path.dirname(__file__)
# Create the full path to the 'fotos' folder
fotos_dir = os.path.join(DIR, './fotos')

features = []
labels = []

# Load the Haar cascade classifier
haar_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

print(fotos_dir)

def createTrain():
    for person in people:
        path = os.path.join(fotos_dir, person)
        label = people.index(person)
        print(path, label, person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Failed to read {img_path}")
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            facesRect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in facesRect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)
                print(f'label = {label}')
                print(f'faces_roi = {faces_roi}')

createTrain()

print(f'features = {features}')
print(f'labels = {labels}')
print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

# Create the LBPH face recognizer
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
# Train the recognizer using the features and labels
faceRecognizer.train(features, labels)
# Save the trained model
faceRecognizer.save(os.path.join(DIR, 'classificadores', 'faceTrained.yml'))
# Save the features and labels arrays
np.save(os.path.join(DIR, './features.npy'), features)
np.save(os.path.join(DIR, './labels.npy'), labels)
