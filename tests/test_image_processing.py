import importlib

import numpy as np

from conv2d_mediana import conv2d_mediana
from grayscale import grayscale
from histograma import histograma
from main import add_noise
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


def test_add_noise_preserves_shape_dtype_and_range():
    image = np.full((8, 8), 127, dtype=np.uint8)

    result = add_noise(image, amount=0.2)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


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
