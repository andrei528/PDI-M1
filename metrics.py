from skimage.metrics import structural_similarity
from sporco.metric import psnr


def _get_ssim_window_size(image):
    min_dimension = min(image.shape)
    if min_dimension >= 7:
        return 7
    if min_dimension % 2 == 0:
        return min_dimension - 1
    return min_dimension


def calculate_metrics(reference_image, comparison_image):
    psnr_value = psnr(reference_image, comparison_image, rng=255)
    ssim_window = _get_ssim_window_size(reference_image)
    ssim_value = structural_similarity(
        reference_image,
        comparison_image,
        data_range=255,
        win_size=ssim_window,
    )

    return {
        "psnr": float(psnr_value),
        "ssim": float(ssim_value),
    }


def print_metrics(image_name, reference_image, processed_images):
    print(f"\nMetricas para {image_name}:")

    for stage_name, image in processed_images.items():
        values = calculate_metrics(reference_image, image)
        print(f"{stage_name}: PSNR={values['psnr']:.4f} | SSIM={values['ssim']:.4f}")


def print_pipeline_metrics(
    image_name,
    grayscale_image,
    noisy_image,
    median_image,
    equalized_image,
    sobel_image,
    unsharp_image,
    highboost_image,
):
    print(f"\nMetricas para {image_name}:")

    stage_pairs = [
        ("ruido vs original", grayscale_image, noisy_image),
        ("mediana vs original", grayscale_image, median_image),
        ("equalizada vs mediana", median_image, equalized_image),
        ("sobel vs equalizada", equalized_image, sobel_image),
        ("unsharp vs sobel", sobel_image, unsharp_image),
        ("highboost vs sobel", sobel_image, highboost_image),
    ]

    for stage_name, reference_image, comparison_image in stage_pairs:
        values = calculate_metrics(reference_image, comparison_image)
        print(f"{stage_name}: PSNR={values['psnr']:.4f} | SSIM={values['ssim']:.4f}")
