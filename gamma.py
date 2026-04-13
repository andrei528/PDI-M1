import numpy as np

def gamma(img):
    c = 0.8
    gamma = 1

    img_out = np.array(c*255*(img/255)**gamma, dtype=np.uint8)
    return img_out