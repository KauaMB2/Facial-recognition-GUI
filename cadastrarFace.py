import cv2
import numpy as np
import tkinter as tk
import time
from tkinter import *
import os

def TirarFotos(nome_funcionario):
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    haarcascade_face = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    amostra = 1
    numeroAmostras = 1000
    largura, altura = 220, 220

    while True:
        _, video = camera.read()
        imagemCinza = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        detectar_face = haarcascade_face.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))

        if not os.path.exists(f'fotos/{nome_funcionario}'):
            os.makedirs(f'fotos/{nome_funcionario}')

        for (x, y, w, h) in detectar_face:
            cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if np.average(imagemCinza) > 30:
                imagemFace = cv2.resize(imagemCinza[y:y + h, x:x + w], (largura, altura))
                dirr = f"fotos/{nome_funcionario}/{amostra}.jpg"
                cv2.imwrite(dirr, imagemFace)
                print("foto " + str(amostra) + " capturada com sucesso!")
                amostra += 1

        cv2.imshow("Video", video)

        if cv2.waitKey(1) == ord("q"):
            break

        if amostra >= numeroAmostras + 1:
            break

    print("Imagens da face capturadas com sucesso!")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    nome_funcionario = input("Digite o nome do funcionário: ")
    if nome_funcionario:
        TirarFotos(nome_funcionario)
