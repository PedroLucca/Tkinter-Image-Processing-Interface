# -- coding: utf-8 --
import cv2
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np

def Abrir():
    global local
    local = filedialog.askopenfilename()
    global img
    img = cv2.imread(local)
    cv2.imshow("Imagem - PDI", img)
def SaveM():
    Salvar(img)
def Salvar(img):
    global local
    cv2.imwrite("Salvo.jpg", img)
def Logaritmica():
    global local
    global img
    imagem = img
    imagem2 = np.uint8(np.log1p(imagem))
    thresh = 204
    # imagem3 = cv2.threshold(imagem2, thresh, 255, cv2.THRESH_TOZERO)[1]
    imagem3 = cv2.threshold(imagem2, thresh, 255, cv2.THRESH_TOZERO_INV)[1]  # Essa Funciona
    # imagem3 = cv2.threshold(imagem2, thresh, 255, cv2.THRESH_TRUNC)[1]
    normalized_image = cv2.normalize(imagem3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imagem4 = cv2.threshold(normalized_image, thresh, 255, cv2.THRESH_TOZERO)[1]
    cv2.imshow("Imagem de entrada", imagem)
    #cv2.imshow("Transformada", imagem3)
    cv2.imshow("Transformada", imagem4)
    cv2.waitKey(0)
def Gama():
    global local
    global img
    def adjust_gamma(gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img, table)
    cv2.imshow('original', img)
    gamma = 0.5  # change the value here to get different result
    adjusted = adjust_gamma(gamma=gamma)
    #cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("gammam image 1", adjusted)
def Inversa():
    global local
    imagem = cv2.imread(local, 0)
    imagem = cv2.bitwise_not(imagem)
    cv2.imshow("Inversa", imagem)
def Exibir():
    global local
    global hist_full
    global img2
    img2 = cv2.imread(local, 0)
    hist_full = cv2.calcHist([img2], [0], None, [256], [0, 256])
    plt.subplot(221), plt.imshow(img2, 'gray')
    plt.subplot(224), plt.plot(hist_full)
    plt.xlim([0, 256])
    plt.show()
def Equalizar():
    global img2
    global hist_equ
    equ = cv2.equalizeHist(img2)
    hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])
    plt.subplot(221), plt.imshow(equ, 'gray')
    plt.subplot(224), plt.plot(hist_equ)
    res = np.hstack((img2, equ))
    cv2.imshow("Interface PDI - Imagem Equalizada", res)
    plt.xlim([0, 256])
    plt.show()
def Comparar():
    Exibir()
def TransformarparaHSV():
    global img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("Imagem HSV", hsv)
def TransformarparaHSI():
    global img
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    cv2.imshow("Imagem HSI", hsl)
def TransformarparaPretoeBranco():
    global local
    img3 = cv2.imread(local, 0)
    (thresh, pb) = cv2.threshold(img3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 87
    pb = cv2.threshold(img3, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Imagem Preto e Branco", pb)
def Mediana():
    global img
    quadro = 3
    img = cv2.medianBlur(img, quadro)
    cv2.imwrite("Mediana.jpg", img)
    cv2.imshow("Filtro da Mediana", img)
def Media():
    global img
    image_corrected = cv2.blur(img, (3, 3))
    cv2.imshow("Filtro da Média", image_corrected)
def MediaPond():
    global img
    image_corrected = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Filtro da Média Ponderada", image_corrected)
def Laplaciano():
    global img
    kernel_size = 3
    result = cv2.Laplacian(img, 0, kernel_size)
    cv2.imshow("Filtro Laplaciano", result)
def RuidoGaussiano():
    global img
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    cv2.imshow("Ruido Gaussiano", noisy)
def RuidoSalePimenta():
    global img
    row, col, ch = img.shape
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
    out[coords] = 0
    cv2.imshow("Ruido Sal e Pimenta", out)
def Erosao():
    global local
    imge = cv2.imread(local, 0)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(imge, kernel, iterations=1)
    cv2.imshow("Imagem Erodida.png", erosion)
def Dilatacao():
    #global local
    global img
    #img = Gray(img)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow("Imagem dilatada.png", img)
def Abertura():
    #global local
    global img
    img = Gray(img)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Abertura.png", opening)
def Fechamento():
    #global local
    global img
    img = Gray(img)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Fechamento.png", img)
def HitorMiss():
    #global local
    global img
    #img = Gray(img)
    kernel = np.ones((3, 3), np.uint8)
    output_image = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
    cv2.imshow("hitormiss", output_image)
def Canny():
    global img
    img_canny = cv2.Canny(img, 100, 200)
    cv2.imshow("Detector de Canny", img_canny)
def Prewitt():
    global local
    gray = cv2.imread(local, 0)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    cv2.imshow("Detector de Prewitt", img_prewittx + img_prewitty)
def LoG():
    global local
    ddepth = cv2.CV_16S
    kernel_size = 3
    gray = cv2.imread(local, 0)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    dst = cv2.Laplacian(img_gaussian, ddepth, kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    cv2.imshow("Laplaciano do Gaussiano", abs_dst)
def Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def Huffman():
    global img
    cv2.imwrite("compress.png", img, cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY)