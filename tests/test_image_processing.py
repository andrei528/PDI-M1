import importlib

import numpy as np

from conv2d_mediana import conv2d_mediana
from grayscale import grayscale
from histograma import histograma
from metrics import calculate_metrics
from unsharp_highboost import highboostFilter, unsharpMask


def test_grayscale_returns_channel_average():
    image = np.array(
        [
            [[30, 60, 90], [0, 30, 60]],
            [[90, 120, 150], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    expected = np.array([[60, 30], [120, 255]], dtype=np.uint8)

    result = grayscale(image)

    assert np.array_equal(result, expected)


def test_histograma_counts_pixel_frequency():
    image = np.array([[0, 0, 1], [2, 2, 2]], dtype=np.uint8)

    result = histograma(image)

    assert result[0] == 2
    assert result[1] == 1
    assert result[2] == 3
    assert result.sum() == image.size


def test_main_module_can_be_imported_without_running_plot():
    module = importlib.import_module("main")

    assert callable(module.main)


def test_unsharp_and_highboost_use_median_filtered_image():
    image = np.array(
        [
            [10, 10, 10],
            [10, 100, 10],
            [10, 10, 10],
        ],
        dtype=np.uint8,
    )

    median_image = conv2d_mediana(image, 3, 3)
    unsharp_image = unsharpMask(image)
    highboost_image = highboostFilter(image)

    assert unsharp_image.shape == image.shape
    assert highboost_image.shape == image.shape
    assert median_image[1, 1] == 10
    assert unsharp_image[1, 1] == 190
    assert highboost_image[1, 1] == 255


def test_metrics_return_psnr_and_ssim():
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )

    result = calculate_metrics(image, image)

    assert "psnr" in result
    assert "ssim" in result
    assert np.isinf(result["psnr"])
    assert result["ssim"] == 1.0
