import cv2
import numpy as np
import tkinter as tk
import time
from tkinter import *
import os

dir_path = os.path.dirname(__file__)

def TirarFotos():
    nome_funcionario = inputName.get()
    if nome_funcionario == "":
        labelRetorno = Label(janela, text="Digite o nome do funcionário!")
        labelRetorno.pack(side=TOP)
        labelRetorno["fg"] = "red"
        return
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    haarcascade_face = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    amostra = 1
    numeroAmostras = 1000
    largura, altura = 220, 220

    cTime=0
    pTime=0
    while True:
        _, video = camera.read()
        imagemCinza = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        detectar_face = haarcascade_face.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))
        cTime=time.time()
        fps=int(1/(cTime-pTime))
        pTime=cTime
        if not os.path.exists(f'fotos/{nome_funcionario}'):
            os.makedirs(f'fotos/{nome_funcionario}')

        for (x, y, w, h) in detectar_face:
            cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if np.average(imagemCinza) > 30:#Verifica se a média dos valores dos pixels da imagem em escala de cinza (imagemCinza) é maior que 100. Essa verificação é usada para garantir que a imagem tenha um brilho suficiente antes de salvar a imagem da face detectada.
                imagemFace = cv2.resize(imagemCinza[y:y + h, x:x + w], (largura, altura))
                dirr = f"fotos/{nome_funcionario}/{amostra}.jpg"
                cv2.imwrite(dirr, imagemFace)

                print("foto " + str(amostra) + " capturada com sucesso!")
                amostra += 1
        cv2.putText(video,f"FPS: {fps}",(5,40),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        cv2.putText(video,f"Amostras: {amostra}/{numeroAmostras}",(5,70),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        cv2.imshow("Video", video)

        if cv2.waitKey(1) == ord("q"):
            break

        if amostra >= numeroAmostras + 1:
            break

    print("Imagens da face capturadas com sucesso!")

    camera.release()
    cv2.destroyAllWindows()

janela = tk.Tk()
janela.geometry("600x600+100+100")

labelTitle = Label(janela, text="Cadastro Facial", font=("calibri 25"))
labelTitle.pack(side=TOP)

labelName = Label(janela, text="Digite o nome do funcionário: ", font=("calibri 12"))
labelName.pack(side=TOP)

inputName = Entry(janela, width=40)
inputName.pack(side=TOP)

buttonRegister = Button(janela, text="Tirar Fotos", width=20, command=TirarFotos)
buttonRegister.pack(side=TOP)
buttonRegister["bg"] = "green"
buttonRegister["fg"] = "white"

janela.mainloop()
