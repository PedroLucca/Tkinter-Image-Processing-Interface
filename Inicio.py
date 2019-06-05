# -- coding: utf-8 --
from tkinter import *
import funcoes

Pdi = Tk()

menubar = Menu(Pdi)
Pdi.config(menu=menubar)
Pdi.title("PDI - Interface Gráfica - Pedro Lucca e Daniel Évani")
Pdi.geometry('1366x768')

menu = Menu(menubar)
menu2 = Menu(menubar)
menu3 = Menu(menubar)
menu4 = Menu(menubar)
menu5 = Menu(menubar)
menu6 = Menu(menubar)
menu8 = Menu(menubar)
menu7 = Menu(menubar)
menu9 = Menu(menubar)

menubar.add_cascade(label='Arquivo', menu=menu)
menubar.add_cascade(label='Realce', menu=menu2)
menubar.add_cascade(label='Histograma', menu=menu3)
menubar.add_cascade(label='Modelos de Cor', menu=menu4)
menubar.add_cascade(label='Filtros Espaciais', menu=menu5)
menubar.add_cascade(label='Inserir Ruído', menu=menu6)
menubar.add_cascade(label='Morfologia Matemática', menu=menu8)
menubar.add_cascade(label='Segmentação', menu=menu7)
menubar.add_cascade(label='Compressão', menu=menu9)


def Sair():
    Pdi.destroy()

menu.add_command(label='Abrir imagem', command=funcoes.Abrir)
menu.add_command(label='Salvar como...', command=funcoes.Salvar)
menu.add_separator()
menu.add_command(label='Sair', command=Sair)
menu2.add_command(label='Logaritmico', command=funcoes.Logaritmica)
menu2.add_command(label='Gama', command=funcoes.Gama)
menu2.add_command(label='Inversa', command=funcoes.Inversa)
menu3.add_command(label='Exibir', command=funcoes.Exibir)
menu3.add_command(label='Equalizar', command=funcoes.Equalizar)
menu3.add_command(label='Comparar', command=funcoes.Comparar)
menu4.add_command(label='Transformar para HSV', command=funcoes.TransformarparaHSV)
menu4.add_command(label='Transformar para HSI', command=funcoes.TransformarparaHSI)
menu4.add_command(label='Transformar para Preto e Branco', command=funcoes.TransformarparaPretoeBranco)
menu5.add_command(label='Filtro da Mediana', command=funcoes.Mediana)
menu5.add_command(label='Filtro de Média', command=funcoes.Media)
menu5.add_command(label='Filtro de Média Ponderada', command=funcoes.MediaPond)
menu5.add_command(label='Filtro Laplaciano', command=funcoes.Laplaciano)
menu6.add_command(label='Ruído Gaussiano', command=funcoes.RuidoGaussiano)
menu6.add_command(label='Ruido Sal e Pimenta', command=funcoes.RuidoSalePimenta)
menu8.add_command(label='Erosão', command=funcoes.Erosao)
menu8.add_command(label='Dilatação', command=funcoes.Dilatacao)
menu8.add_command(label='Abertura', command=funcoes.Abertura)
menu8.add_command(label='Fechamento', command=funcoes.Fechamento)
menu8.add_command(label='Hit-or-Miss', command=funcoes.HitorMiss)
menu7.add_command(label='Detector de Canny', command=funcoes.Canny)
menu7.add_command(label='Detector de Prewitt', command=funcoes.Prewitt)
menu7.add_command(label='Laplaciano da Gaussiana', command=funcoes.LoG)
menu9.add_command(label='Huffman', command=funcoes.Huffman)


Pdi.mainloop()