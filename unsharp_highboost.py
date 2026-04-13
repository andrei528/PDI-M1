import numpy as np

from conv2d_mediana import conv2d_mediana

def unsharpMask(img, kernel=None, k=1, sigma=1, size_x=3, size_y=3):
    img_uint8 = np.array(img, dtype=np.uint8)
    img_float = np.array(img, dtype=np.float32)
    img_blur = np.array(conv2d_mediana(img_uint8, size_x, size_y, True), dtype=np.float32)

    img_height, img_width = img_float.shape
    output = np.zeros((img_height, img_width), dtype=np.uint8)

    for i in range(img_height):
        for j in range(img_width):
            mask = img_float[i, j] - img_blur[i, j]
            pixel = img_float[i, j] + (k * mask)

            if pixel > 255:
                pixel = 255
            elif pixel < 0:
                pixel = 0

            output[i, j] = int(pixel)

    return output

def highboostFilter(img, kernel=None, k=2, sigma=1, size_x=3, size_y=3):
    return unsharpMask(img, kernel=kernel, k=k, sigma=sigma, size_x=size_x, size_y=size_y)
