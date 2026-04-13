import cv2
import numpy as np


def grayscale(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    img_grayscale = (0.114 * b_channel) + (0.587 * g_channel) + (0.299 * r_channel)
    return np.array(img_grayscale, dtype=np.uint8)
