import sys
import os
from contextlib import contextmanager

from SSIM_PIL import compare_ssim
from PIL import Image
import sporco.metric as sm

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def calculate_metrics(reference_image, comparison_image):
    with suppress_stdout():
        reference_image = reference_image.astype('uint8')
        comparison_image = comparison_image.astype('uint8')

        psnr_value = sm.psnr(reference_image, comparison_image)
        
        image1 = Image.fromarray(reference_image)
        image2 = Image.fromarray(comparison_image)
        ssim_value = compare_ssim(image1, image2)

    print(f"PSNR:  {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print("-" * 9)
