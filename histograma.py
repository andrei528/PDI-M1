import numpy as np


def histograma(img):
    histogram = np.zeros(256)
    x_size, y_size = img.shape

    for i in range(x_size):
        for j in range(y_size):
            histogram[img[i, j]] += 1

    return histogram


# Equaliza o histograma
def histogram_equalized(histogram, img):
    cdf = np.zeros(256)
    cdf[0] = histogram[0]

    x_size, y_size = img.shape

    # Faz a soma acumulativa
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    # Encontrar o primeiro valor não 0
    cdf_min = 0
    for i in range(256):
        if cdf[i] != 0:
            cdf_min = cdf[i]
            break

    total_pixels = x_size * y_size

    # Normalizar histograma
    cdf_normalized = np.zeros(256)

    for i in range(256):
        cdf_normalized[i] = ((cdf[i] - cdf_min) * 255) / (total_pixels - cdf_min)

    cdf_normalized = np.array(cdf_normalized, dtype=np.uint8)

    return cdf_normalized


def img_equalized(histogram_equalized, img):

    x_size, y_size = img.shape
    img_eq = np.zeros((x_size, y_size), dtype=np.uint8)

    for i in range(x_size):
        for j in range(y_size):
            img_eq[i, j] = histogram_equalized[img[i, j]]

    return img_eq
