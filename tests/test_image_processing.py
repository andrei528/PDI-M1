import importlib

import numpy as np

from grayscale import grayscale
from histograma import histograma


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
