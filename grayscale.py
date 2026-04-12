import cv2
import numpy as np


def grayscale(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    img_grayscale = (b_channel / 3) + (g_channel / 3) + (r_channel / 3)
    return np.array(img_grayscale, dtype=np.uint8)
