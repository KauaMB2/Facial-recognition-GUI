import cv2 as cv
import os
import sys
from FancyDrawn import FancyDrawn


class ReconhecimentoFacial():
    def __init__(self, image_path):
        self.__haar_cascade = cv.CascadeClassifier(r'cascades\haarcascade_frontalface_default.xml')
        self.__people = os.listdir("fotos")
        self.__faceRecognizer = cv.face.LBPHFaceRecognizer_create()
        self.__faceRecognizer.read(r'classificadores\faceTrained.yml')
        self.__img = cv.imread(image_path)
    def reconhecer(self):
        gray = cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)
        facesRect = self.__haar_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in facesRect:
            facesRoi = gray[y:y + h, x:x + w]
            label, confidence = self.__faceRecognizer.predict(facesRoi)
            if len(self.__people) != 0:
                print(f'Label = {self.__people[label]} with a confidence of {confidence}')
                text = f"{self.__people[label]} - {round(confidence, 2)}%"
                (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 0.9, 4)
                cv.rectangle(self.__img, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
                cv.putText(self.__img, text, (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), thickness=2)
                fancydrawn=FancyDrawn(self.__img)
                self.__img=fancydrawn.draw((x, y, w, h))
        cv.imshow('Detected Face', self.__img)
        cv.waitKey(0)
        cv.destroyAllWindows()
