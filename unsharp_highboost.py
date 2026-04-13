import numpy as np

from gaussian_filter import gauss_create
from conv2d import conv2d


def highBoost(img, k, imgSobel):

    img = np.array(img, dtype=np.float32)
    output = img + (k * imgSobel)

    imgH, imgW = img.shape
    for i in range(imgH):
        for j in range(imgW):
            if output[i, j] > 255:
                output[i, j] = 255
            elif output[i, j] < 0:
                output[i, j] = 0

    return np.array(output, dtype=np.uint8)
