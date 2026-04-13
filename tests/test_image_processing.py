import importlib

import numpy as np

from conv2d_mediana import conv2d_mediana
from grayscale import grayscale
from histograma import histograma
from metrics import calculate_metrics
from unsharp_highboost import highBoost


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


def test_median_filter_reduces_center_impulse():
    image = np.array(
        [
            [10, 10, 10],
            [10, 100, 10],
            [10, 10, 10],
        ],
        dtype=np.uint8,
    )

    result = conv2d_mediana(image, 3, 3)

    assert result.shape == image.shape
    assert result[1, 1] == 10


def test_highboost_combines_input_with_edge_image():
    image = np.array(
        [
            [10, 10, 10],
            [10, 100, 10],
            [10, 10, 10],
        ],
        dtype=np.uint8,
    )
    edge_image = np.array(
        [
            [0, 0, 0],
            [0, 90, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = highBoost(image, 1, edge_image)

    assert result.shape == image.shape
    assert result[1, 1] == 190


def test_highboost_clips_values_above_255():
    image = np.array([[100]], dtype=np.uint8)
    edge_image = np.array([[90]], dtype=np.uint8)

    result = highBoost(image, 2, edge_image)

    assert result[0, 0] == 255


def test_metrics_prints_psnr_and_ssim(capsys):
    image = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
        dtype=np.uint8,
    )

    result = calculate_metrics(image, image)
    captured = capsys.readouterr()

    assert result is None
    assert "PSNR:" in captured.out
    assert "SSIM:" in captured.out
