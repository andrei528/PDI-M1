import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise

from conv2d_mediana import conv2d_mediana
from grayscale import grayscale
from histograma import histograma, histogram_equalized, img_equalized
from unsharp_highboost import highBoost
from sobel import sobel
from metrics import calculate_metrics
from gamma import gamma


def load_images():
    img_blood = cv2.imread("Images/imagem_Blood.png")

    img_organs = cv2.imread("Images/imagem_Organs.png")

    img_retina = cv2.imread("Images/imagem_Retina.png")

    return img_blood, img_organs, img_retina


def add_noise(image):
    noisy_image = random_noise(image / 255, mode="s&p")
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


def process_image(image, gammaTrue = False):
    grayscale_image = grayscale(image)
    noisy_image = add_noise(grayscale_image)
    median_image = conv2d_mediana(noisy_image, 3, 3)
    if gammaTrue:
        contrast_image = gamma(median_image)
    else:
        contrast_image = img_equalized(
            histogram_equalized(histograma(median_image), median_image), median_image
        )
    sobel_image = sobel(contrast_image)
    final_image = highBoost(contrast_image, 0.5, sobel_image)

    return (
        grayscale_image,
        noisy_image,
        median_image,
        contrast_image,
        sobel_image,
        final_image,
    )


def main():
    img_blood, img_organs, img_retina = load_images()

    (
        img_grayscale_blood,
        img_blood_ruido,
        img_blood_mediana,
        img_blood_contrast,
        img_blood_sobel,
        img_blood_final,
    ) = process_image(img_blood)
    (
        img_grayscale_organs,
        img_organs_ruido,
        img_organs_mediana,
        img_organs_contrast,
        img_organs_sobel,
        img_organs_final,
    ) = process_image(img_organs)
    (
        img_grayscale_retina,
        img_retina_ruido,
        img_retina_mediana,
        img_retina_contrast,
        img_retina_sobel,
        img_retina_final,
    ) = process_image(img_retina)

    """
    plot_histogram(img_grayscale_blood)
    plot_histogram(img_grayscale_organs)
    plot_histogram(img_grayscale_retina)

    plot_histogram(img_blood_mediana)
    plot_histogram(img_organs_mediana)
    plot_histogram(img_retina_mediana)
    """
    print("BLOOD")
    calculate_metrics(img_grayscale_blood, img_blood_final)
    
    print("")
    print("ORGANS")
    calculate_metrics(img_grayscale_organs, img_organs_final)
    
    print("")
    print("RETINA")
    calculate_metrics(img_grayscale_retina, img_retina_final)
    
    #print(" organs ")
    #calculate_metrics(img_grayscale_organs, img_organs_ruido)
    #calculate_metrics(img_grayscale_organs, img_organs_final)
    
    #print(" RETINA ")
    #calculate_metrics(img_grayscale_retina, img_retina_ruido)
    #calculate_metrics(img_grayscale_retina, img_retina_final)

    #cv2.imshow("imgBlood", img_blood)
    #cv2.imshow("imgGrayScaleBlood", img_grayscale_blood)
    #cv2.imshow("imgRuidoBlood", img_blood_ruido)
    #cv2.imshow("imgMedianaBlood", img_blood_mediana)
    #cv2.imshow("contrastBlood", img_blood_contrast)
    #cv2.imshow("sobelBlood", img_blood_sobel)
    cv2.imshow("highboostBlood", img_blood_final)
    
    #cv2.imshow("imgOrgans", img_organs)
    #cv2.imshow("imgGrayScaleOrgans", img_grayscale_organs)
    #cv2.imshow("imgRuidoOrgans", img_organs_ruido)
    #cv2.imshow("imgMedianaOrgans", img_organs_mediana)
    #cv2.imshow("contrastOrgans", img_organs_contrast)
    #cv2.imshow("sobelOrgans", img_organs_sobel)
    cv2.imshow("highboostOrgans", img_organs_final)
    
    #cv2.imshow("imgRetina", img_retina)
    #cv2.imshow("imgGrayScaleRetina", img_grayscale_retina)
    #cv2.imshow("imgRuidoRetina", img_retina_ruido)
    #cv2.imshow("imgMedianaRetina", img_retina_mediana)
    #cv2.imshow("contrastRetina", img_retina_contrast)
    #cv2.imshow("sobelRetina", img_retina_sobel)
    cv2.imshow("highboostRetina", img_retina_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    cv2.imshow("in", img_organs_mediana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("in", img_retina_mediana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


if __name__ == "__main__":
    main()
