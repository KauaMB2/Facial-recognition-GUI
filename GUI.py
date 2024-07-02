import tkinter as tk
import threading
import os
import subprocess
import cv2
import numpy as np
import time
from tkinter import filedialog


# Global variables
labelRetorno = None

def run_file(filename, file_path=None):
    if filename == 'treinamento.py':
        start_training()
    elif filename == 'reconhecimentoFacial.py' and file_path:
        subprocess.run(['python', filename, file_path])
    else:
        subprocess.run(['python', filename])

def start_training():
    # Display training message
    training_label.config(text="A inteligência artificial está sendo treinada, por favor aguarde...", fg="white")
    training_label.pack(pady=20)
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_ai)
    training_thread.start()

def train_ai():
    import treinamento

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        run_file('reconhecimentoFacial.py', file_path)

def TirarFotos(nome_funcionario):
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
        cv2.putText(video,f"FPS: {fps}",(5,40),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        cv2.putText(video,f"Amostras: {amostra}/{numeroAmostras}",(5,70),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        for (x, y, w, h) in detectar_face:
            cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if np.average(imagemCinza) > 30:
                imagemFace = cv2.resize(imagemCinza[y:y + h, x:x + w], (largura, altura))
                dirr = f"fotos/{nome_funcionario}/{amostra}.jpg"
                cv2.imwrite(dirr, imagemFace)
                print("foto " + str(amostra) + " capturada com sucesso!")
                amostra += 1
                print("foto " + str(amostra) + " capturada com sucesso!")
        cv2.imshow("Video", video)

        if cv2.waitKey(1) == ord("q"):
            break

        if amostra >= numeroAmostras + 1:
            break

    print("Imagens da face capturadas com sucesso!")
    camera.release()
    cv2.destroyAllWindows()

def cadastrar_funcionario():
    global labelRetorno  # Declare labelRetorno as global

    nome_funcionario = inputName.get()

    if labelRetorno:
        labelRetorno.pack_forget()  # Remove the label if it's currently visible
    
    if nome_funcionario == "":
        labelRetorno = tk.Label(janela, text="Digite o nome do usuário!", fg="red")
        labelRetorno.pack(side=tk.TOP)
        return
    
    TirarFotos(nome_funcionario)
    

janela = tk.Tk()
janela.geometry("400x300")
janela.configure(bg="#6a0dad")  # Cor de fundo roxo

labelTitle = tk.Label(janela, text="Reconhecimento Facial", font=("calibri", 30), bg="#6a0dad", fg="white")
labelTitle.pack(side=tk.TOP)

labelName = tk.Label(janela, text="Digite o nome da usuário: ", font=("calibri", 12), bg="#6a0dad", fg="white")
labelName.pack(side=tk.TOP)

inputName = tk.Entry(janela, width=40)
inputName.pack(side=tk.TOP)

buttonRegister = tk.Button(janela, text="Cadastrar nova face", width=20, command=cadastrar_funcionario, bg="#9370DB", fg="white")
buttonRegister.pack(side=tk.TOP)

buttons_frame = tk.Frame(janela, bg="#6a0dad")
buttons_frame.pack(side=tk.TOP)

button_camera = tk.Button(buttons_frame, text="Abrir Câmera", width=30, command=lambda: run_file('camera.py'), bg="#9370DB", fg="white")
button_camera.pack(pady=10)

button_treinamento = tk.Button(buttons_frame, text="Treinar IA", width=30, command=lambda: run_file('treinamento.py'), bg="#9370DB", fg="white")
button_treinamento.pack(pady=10)

button_reconhecimento = tk.Button(buttons_frame, text="Reconhecer em Imagem", width=30, command=open_image, bg="#9370DB", fg="white")
button_reconhecimento.pack(pady=10)

training_label = tk.Label(janela, text="", font=("calibri", 12), bg="#483D8B")

janela.mainloop()
