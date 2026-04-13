import numpy as np


def histograma(img):
    histogram = np.zeros(256)
    x_size, y_size = img.shape

    for i in range(x_size):
        for j in range(y_size):
            histogram[img[i, j]] += 1

    return histogram
