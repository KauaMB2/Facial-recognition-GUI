import tkinter as tk
from tkinter import *
import os
from tkinter import filedialog
from CadastrarFace import CadastrarFace
from Treinamento import Treinamento
from Camera import Camera
from ReconhecimentoFacial import ReconhecimentoFacial

# Global variables
labelRetorno = None

def runCamera():
    camera=Camera()
    camera.reconhecer()

def start_training():
    treinamento=Treinamento()
    treinamento.createTrain()

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        reconhecimento=ReconhecimentoFacial(file_path)
        reconhecimento.reconhecer()

def cadastrarUsuario():
    global labelRetorno  # Declare labelRetorno as global
    nomeUsuario = inputName.get()
    if labelRetorno:
        labelRetorno.pack_forget()  # Remove the label if it's currently visible
    if nomeUsuario == "":
        labelRetorno = tk.Label(janela, text="Digite o nome do usuário!", fg="red")
        labelRetorno.pack(side=tk.TOP)
        return
    cadastrarFace=CadastrarFace(nomeUsuario)
    cadastrarFace.TirarFotos()

janela = tk.Tk()
img = PhotoImage(file=os.path.join(os.path.dirname(__file__), 'icon', 'icon.png'))
janela.iconphoto(False, img)
janela.title("Reconhecimento facial")
janela.geometry("400x300")
janela.configure(bg="#6a0dad")  # Cor de fundo roxo

labelTitle = tk.Label(janela, text="Reconhecimento Facial", font=("calibri", 30), bg="#6a0dad", fg="white")
labelTitle.pack(side=tk.TOP)

labelName = tk.Label(janela, text="Digite o nome da usuário: ", font=("calibri", 12), bg="#6a0dad", fg="white")
labelName.pack(side=tk.TOP)

inputName = tk.Entry(janela, width=40)
inputName.pack(side=tk.TOP)

buttonRegister = tk.Button(janela, text="Cadastrar nova face", width=20, command=cadastrarUsuario, bg="#9370DB", fg="white")
buttonRegister.pack(side=tk.TOP)

buttons_frame = tk.Frame(janela, bg="#6a0dad")
buttons_frame.pack(side=tk.TOP)

button_camera = tk.Button(buttons_frame, text="Abrir Câmera", width=30, command=runCamera, bg="#9370DB", fg="white")
button_camera.pack(pady=10)

button_treinamento = tk.Button(buttons_frame, text="Treinar IA", width=30, command=start_training, bg="#9370DB", fg="white")
button_treinamento.pack(pady=10)

button_reconhecimento = tk.Button(buttons_frame, text="Reconhecer em Imagem", width=30, command=open_image, bg="#9370DB", fg="white")
button_reconhecimento.pack(pady=10)

training_label = tk.Label(janela, text="", font=("calibri", 12), bg="#483D8B")

janela.mainloop()
