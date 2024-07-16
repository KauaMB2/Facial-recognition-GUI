import cv2 as cv
import os
import sys
from FancyDrawn import FancyDrawn

if len(sys.argv) < 2:
    print("Usage: python reconhecimentoFacial.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

haar_cascade = cv.CascadeClassifier(r'cascades\haarcascade_frontalface_default.xml')

people = os.listdir("fotos")
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r'classificadores\faceTrained.yml')

img = cv.imread(image_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
facesRect = haar_cascade.detectMultiScale(gray, 1.1, 4)

def fancyDraw(frame, bbox, l=30, t=10):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    # Top Left
    cv.line(frame, (x, y), (x + l, y), (0, 255, 0), t)
    cv.line(frame, (x, y), (x, y + l), (0, 255, 0), t)
    # Top Right
    cv.line(frame, (x1, y), (x1 - l, y), (0, 255, 0), t)
    cv.line(frame, (x1, y), (x1, y + l), (0, 255, 0), t)
    # Bottom Left
    cv.line(frame, (x, y1), (x + l, y1), (0, 255, 0), t)
    cv.line(frame, (x, y1), (x, y1 - l), (0, 255, 0), t)
    # Bottom Right
    cv.line(frame, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
    cv.line(frame, (x1, y1), (x1, y1 - l), (0, 255, 0), t)

for (x, y, w, h) in facesRect:
    facesRoi = gray[y:y + h, x:x + w]
    label, confidence = faceRecognizer.predict(facesRoi)
    if len(people) != 0:
        print(f'Label = {people[label]} with a confidence of {confidence}')
        text = f"{people[label]} - {round(confidence, 2)}%"
        (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 0.9, 4)
        cv.rectangle(img, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)
        fancydrawn=FancyDrawn(img)
        img=fancydrawn.draw((x, y, w, h))

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
