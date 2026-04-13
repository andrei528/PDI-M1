import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise

from conv2d_mediana import conv2d_mediana
from grayscale import grayscale
from histograma import histograma, histogram_equalized, img_equalized
from unsharp_highboost import highboostFilter, unsharpMask
from sobel import sobel

def load_images():
    img_blood = cv2.imread("Images/imagem_Blood.png")
    if img_blood is None:
        raise FileNotFoundError("Nao foi possivel carregar Images/imagem_Blood.png")
    img_blood = cv2.resize(img_blood, (256, 256))

    img_path = cv2.imread("Images/imagem_Path.png")
    if img_path is None:
        raise FileNotFoundError("Nao foi possivel carregar Images/imagem_Path.png")
    img_path = cv2.resize(img_path, (256, 256))

    img_retina = cv2.imread("Images/imagem_Retina.png")
    if img_retina is None:
        raise FileNotFoundError("Nao foi possivel carregar Images/imagem_Retina.png")
    img_retina = cv2.resize(img_retina, (256, 256))

    return img_blood, img_path, img_retina


def add_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy_image = random_noise(
        image / 255,
        mode="s&p",
        amount=amount,
        salt_vs_pepper=salt_vs_pepper,
    )
    noisy_image = noisy_image * 255
    return np.array(noisy_image, dtype=np.uint8)


def plot_histogram(image):
    histogram = histograma(image)
    x = np.linspace(0, 255, 256)
    plt.bar(x, histogram)
    plt.xlabel("intensidade")
    plt.ylabel("frequência")
    plt.show()

    return histogram


def process_image(image):
    grayscale_image = grayscale(image)
    noisy_image = add_noise(grayscale_image)
    median_image = conv2d_mediana(noisy_image, 3, 3)
    equalized_image = img_equalized(histogram_equalized(histograma(median_image), median_image), median_image)
    sobel_image = sobel(equalized_image)
    unsharp_image = unsharpMask(sobel_image)
    highboost_image =highboostFilter(unsharp_image)

    return grayscale_image, noisy_image, median_image, equalized_image, sobel_image, unsharp_image, highboost_image


def main():
    img_blood, img_path, img_retina = load_images()

    img_grayscale_blood, img_blood_ruido, img_blood_mediana, img_blood_equalized, img_blood_sobel, img_blood_unsharp, img_blood_highboost = process_image(img_blood)
    img_grayscale_path, img_path_ruido, img_path_mediana, img_path_equalized, img_path_sobel, img_path_unsharp, img_path_highboost = process_image(img_path)
    img_grayscale_retina, img_retina_ruido, img_retina_mediana, img_retina_equalized, img_retina_sobel, img_retina_unsharp, img_retina_highboost = process_image(img_retina)

    """
    plot_histogram(img_grayscale_blood)
    plot_histogram(img_grayscale_path)
    plot_histogram(img_grayscale_retina)

    plot_histogram(img_blood_mediana)
    plot_histogram(img_path_mediana)
    plot_histogram(img_retina_mediana)
    """

    cv2.imshow("imgRuido", img_blood_ruido)
    cv2.imshow("equalizada", img_blood_equalized)
    cv2.imshow("sobel", img_blood_sobel)
    cv2.imshow("unsharp", img_blood_unsharp)
    cv2.imshow("highboost", img_blood_highboost)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    cv2.imshow("in", img_path_mediana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("in", img_retina_mediana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


if __name__ == "__main__":
    main()
