import numpy as np


def add_padding(img, padding_height, padding_width):
    n, m = img.shape
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img

    return padded_img


def prepare_convolution(img, kernel_height, kernel_width, padding=True):
    img_height, img_width = img.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    if padding:
        padded_img = add_padding(img, pad_height, pad_width)
    else:
        padded_img = img

    output = np.zeros((img_height, img_width), dtype=float)

    return padded_img, output, img_height, img_width
