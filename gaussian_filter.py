import numpy as np

from conv2d import conv2d


def gauss_create(sigma=1, size_x=3, size_y=3):
    """
    Create normal (gaussian) distribuiton
    """

    x, y = np.meshgrid(np.linspace(-1, 1, size_x), np.linspace(-1, 1, size_y))
    calc = 1 / ((2 * np.pi * (sigma**2)))
    exp = np.exp(-(((x**2) + (y**2)) / (2 * (sigma**2))))

    return exp * calc


def gaussian_filter(img, sigma=1, size_x=3, size_y=3):
    kernel = gauss_create(sigma=sigma, size_x=size_x, size_y=size_y)
    kernel_sum = 0.0

    for i in range(size_y):
        for j in range(size_x):
            kernel_sum += kernel[i, j]

    kernel = kernel / kernel_sum

    return conv2d(img, kernel)
