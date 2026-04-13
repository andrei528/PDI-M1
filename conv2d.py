import numpy as np

from padding import prepare_convolution


def conv2d(img, kernel, padding=True):
    k_height, k_width = kernel.shape
    padded_img, output, img_height, img_width = prepare_convolution(img, k_height, k_width, padding)

    for i in range(img_height):
        for j in range(img_width):
            soma = 0
            for k in range(k_height):
                for l in range(k_width):
                    soma += padded_img[i + k, j + l] * kernel[k, l]

            if soma > 255:
                soma = 255
            if soma < 0:
                soma = 0
            output[i, j] = soma

    return np.array(output, dtype=np.uint8)
