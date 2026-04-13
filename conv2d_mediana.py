import numpy as np

from conv2d import add_padding


def conv2d_mediana(img, kernel_w, kernel_h, padding=True):
    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = kernel_h // 2  # Atribui valor à variável pad_height
    pad_width = kernel_w // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(
            img, pad_height, pad_width
        )  # Atribui valor à variável padded_img

    # print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros(
        (img_height, img_width), dtype=float
    )  # Atribui valor à variável output

    # Perform convolution
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

            mediana = vet[len(vet) // 2]

            output[i, j] = mediana

    return np.array(output, dtype=np.uint8)
