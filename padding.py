import numpy as np

def add_padding(img, padding_height, padding_width):
    n, m = img.shape
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img

    return padded_img