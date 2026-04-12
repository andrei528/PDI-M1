import cv2
import numpy as np

def grayscale(img):
    # Transformando em escala de cinza
    B, G, R = cv2.split(img)
    img_grayscale = (B/3)+(G/3)+(R/3)
    img_grayscale = np.array(img_grayscale, dtype=np.uint8)
    return img_grayscale