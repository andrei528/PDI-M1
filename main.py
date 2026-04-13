import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise

from grayscale import grayscale
from histograma import histograma, histogram_equalized, img_equalized

# Carrega e altera tamanho da imagem para 256x256
def load_images():
    img_blood = cv2.imread("Images/imagem_Blood.png")
    img_blood = cv2.resize(img_blood, (256, 256))

    img_path = cv2.imread("Images/imagem_Path.png")
    img_path = cv2.resize(img_path, (256, 256))

    img_retina = cv2.imread("Images/imagem_Retina.png")
    img_retina = cv2.resize(img_retina, (256, 256))

    return img_blood, img_path, img_retina

# Adiciona ruído na imagem
def add_noise(image):
    noisy_image = random_noise(image / 255, mode="s&p")
    noisy_image = noisy_image * 255
    return np.array(noisy_image, dtype=np.uint8)

# Cria histograma da imagem
def plot_histogram(image):
    histogram = histograma(image)
    #x = np.linspace(0, 255, 256)
    #plt.bar(x, histogram)
    #plt.xlabel("intensidade")
    #plt.ylabel("frequência")
    #plt.show()
    
    histogram = histogram_equalized(histogram, image)
    
    #plt.bar(x, histogram)
    #plt.xlabel("intensidade")
    #plt.ylabel("frequência")
    #plt.show()
    img_eq = img_equalized( histogram, image)
    
    return img_eq

def main():
    img_blood, img_path, img_retina = load_images()

    img_grayscale_blood = grayscale(img_blood)
    img_grayscale_path = grayscale(img_path)
    img_grayscale_retina = grayscale(img_retina)

    img_blood_ruido = add_noise(img_grayscale_blood)
    img_path_ruido = add_noise(img_grayscale_path)
    img_retina_ruido = add_noise(img_grayscale_retina)

    img_equalized = plot_histogram(img_blood_ruido)
    
    cv2.imshow('bloodEqualized', img_equalized)
    cv2.imshow('blood', img_blood_ruido)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plot_histogram(img_path_ruido)
    #plot_histogram(img_retina_ruido)
    
    

if __name__ == "__main__":
    main()
