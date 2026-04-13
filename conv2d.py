import numpy as np

from padding import add_padding

def conv2d(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img

    #print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output

    # Perform convolution
    for i in range(img_height):
      for j in range(img_width):
        soma = 0
        for k in range(k_height):
          for l in range(k_width):
            soma += padded_img[i+k, j+l] * kernel[k, l]
            
        if soma > 255:
          soma = 255
        if soma < 0:
          soma = 0
        output[i, j] = soma
        
    return np.array(output, dtype=np.uint8)