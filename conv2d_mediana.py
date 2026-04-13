import numpy as np

from padding import prepare_convolution


def conv2d_mediana(img, kernel_w, kernel_h, padding=True):
    padded_img, output, img_height, img_width = prepare_convolution(img, kernel_h, kernel_w, padding)

    for i in range(img_height):
        for j in range(img_width):
            vet = np.zeros(kernel_h * kernel_w)
            idx = 0
            for k in range(kernel_h):
                for l in range(kernel_w):
                    vet[idx] = padded_img[k + i, l + j]
                    idx += 1

            for a in range(len(vet)):
                for b in range(len(vet) - 1):
                    if vet[b] > vet[b + 1]:
                        aux = vet[b]
                        vet[b] = vet[b + 1]
                        vet[b + 1] = aux

            output[i, j] = vet[len(vet) // 2]

    return np.array(output, dtype=np.uint8)
