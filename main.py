import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

from grayscale import grayscale
from histograma import histograma


# Carregando as imagens
img_blood = cv2.imread("Images/imagem_Blood.png")
img_blood = cv2.resize(img_blood, (256, 256))

img_path = cv2.imread("Images/imagem_Path.png")
img_path = cv2.resize(img_path, (256, 256))

img_retina = cv2.imread("Images/imagem_Retina.png")
img_retina = cv2.resize(img_retina, (256, 256))

# Transformando em escala de cinza
img_grayscale_blood = grayscale(img_blood)

img_grayscale_path = grayscale(img_path)

img_grayscale_retina = grayscale(img_retina)

# Aplicando ruído Sal e Pimenta

img_blood_ruido = random_noise(img_grayscale_blood/255, mode="s&p")
img_blood_ruido = img_blood_ruido * 255
img_blood_ruido = np.array(img_blood_ruido, dtype=np.uint8)

img_path_ruido = random_noise(img_grayscale_path/255, mode="s&p")
img_path_ruido = img_path_ruido * 255
img_path_ruido = np.array(img_path_ruido, dtype=np.uint8)

img_retina_ruido = random_noise(img_grayscale_retina/255, mode="s&p")
img_retina_ruido = img_retina_ruido * 255
img_retina_ruido = np.array(img_retina_ruido, dtype=np.uint8)

# Mostrando as imagens
#cv2.imshow('blood', img_grayscale_blood)
#cv2.imshow('path', img_grayscale_path)
#cv2.imshow('retina', img_grayscale_retina)
#cv2.imshow('bloodRuido', img_blood_ruido)
#cv2.imshow('pathRuido', img_path_ruido)
#cv2.imshow('retinaRuido', img_retina_ruido)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

histogram = histograma(img_blood_ruido)

x = np.linspace(0, 255, 256)
plt.bar(x, histogram)
plt.xlabel('intensidade');
plt.ylabel('frequência');
plt.show()