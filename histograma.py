import numpy as np
import matplotlib.pyplot as plt

def histograma(img):
    
    histogram = np.zeros(256)
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            histogram[img[i, j]] += 1
            
    return histogram