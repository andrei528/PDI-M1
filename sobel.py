from conv2d import conv2d

import numpy as np

def sobel(img):
    img = np.array(img, dtype=np.float32)

    kernelGx = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1] ])

    kernelGy = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1] ])

    gx = conv2d(img, kernelGx)
    gy = conv2d(img, kernelGy)

    imgH, imgW = img.shape

    output = np.zeros((imgH, imgW), dtype=float)
    
    max_val = 0

    for i in range(imgH):
        for j in range(imgW):
            result = abs(float(gx[i, j])) + abs(float(gy[i, j]))
            if result > max_val:
                max_val = result
            output[i, j] = result
    
    if max_val > 0:
        output = (output / max_val) * 255
        
    output = np.array(output, dtype=np.float32) + np.array(img, dtype=np.float32)
    for i in range(256):
        for j in range(256):
            if output[i, j] > 255:
                output[i, j] = 255

    return np.array(output, dtype=np.uint8)