from skimage.metrics import structural_similarity
from sporco.metric import gmsd, psnr


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
    gmsd_value = gmsd(reference_image, comparison_image)

    return {
        "psnr": float(psnr_value),
        "ssim": float(ssim_value),
        "gmsd": float(gmsd_value),
    }


def _print_block(title, reference_image, stage_pairs):
    print()
    print(title)
    for stage_name, comparison_image in stage_pairs:
        values = calculate_metrics(reference_image, comparison_image)
        print(f"{stage_name}: PSNR={values['psnr']:.4f} | SSIM={values['ssim']:.4f} | GMSD={values['gmsd']:.4f}")


def print_metrics(image_name, reference_image, processed_images):
    print()
    print("Metricas para " + image_name + ":")

    for stage_name, image in processed_images.items():
        values = calculate_metrics(reference_image, image)
        print(f"{stage_name}: PSNR={values['psnr']:.4f} | SSIM={values['ssim']:.4f} | GMSD={values['gmsd']:.4f}")


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
    vs_clean = [
        ("ruido", noisy_image),
        ("mediana", median_image),
        ("equalizada", equalized_image),
        ("unsharp", unsharp_image),
        ("highboost", highboost_image),
        ("sobel (bordas)", sobel_image),
    ]
    _print_block(
        "Metricas para " + image_name + " (referencia: grayscale limpa):",
        grayscale_image,
        vs_clean,
    )

    vs_noisy = [
        ("mediana", median_image),
        ("equalizada", equalized_image),
        ("unsharp", unsharp_image),
        ("highboost", highboost_image),
        ("sobel (bordas)", sobel_image),
    ]
    _print_block(
        "Metricas para " + image_name + " (referencia: imagem com ruido):",
        noisy_image,
        vs_noisy,
    )
