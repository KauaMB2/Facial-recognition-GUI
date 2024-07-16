import cv2 as cv
import time
import os
import numpy as np
from FancyDrawn import FancyDrawn  # Corrigir importação

haar_cascade = cv.CascadeClassifier(r'cascades\haarcascade_frontalface_default.xml')
people = os.listdir("fotos")  # Coloca o nome de todos as pastas dentro da pasta "fotos" em uma lista
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r'classificadores\faceTrained.yml')

camera = cv.VideoCapture(0, cv.CAP_DSHOW)
#Time variables
cTime = 0
pTime = 0

while camera.isOpened():
    status, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    facesRect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    #Show FPS
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv.putText(frame, f"FPS: {fps}", (5, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) #putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)

    for (x, y, w, h) in facesRect:
        facesRoi = gray[y:y+h, x:x+h]
        label, confidence = faceRecognizer.predict(facesRoi)
        if len(people) != 0:
            print(f'Label = {people[label]} with a confidence of {confidence}')
            text = f"{people[label]} - {round(confidence, 2)}%"
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 0.9, 4) # Get the width and height of the text box
            cv.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1) # Draw the rectangle background for the text
            cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), thickness=2) # Put the text on top of the rectangle
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4) # Draw the bounding box
            fancydraw = FancyDrawn(frame)
            frame = fancydraw.draw((x, y, w, h)) # Draw the fancy bounding box

    key = cv.waitKey(1)  # ESC = 27
    if key == 27:  # Se apertou o ESC
        break
    
    cv.imshow("Camera", frame)

camera.release()
cv.destroyAllWindows()
