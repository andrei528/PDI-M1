import numpy as np


def gamma_correction(img, gamma=1.2):
    img_float = np.array(img, dtype=np.float32) / 255.0
    corrected = np.power(img_float, gamma)
    return np.array(corrected * 255, dtype=np.uint8)
