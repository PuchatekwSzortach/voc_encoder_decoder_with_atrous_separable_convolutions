"""
Tests for net.processing module
"""

import numpy as np

import net.processing


def test_get_sparse_segmentation_labels_image():
    """
    Test for get_sparse_segmentation_labels_image that changes segmentation image into a 2D array with
    segmentation categories set for corresponding pixels values
    """

    void_color = [100, 100, 100]

    indices_to_colors_map = {
        1: [10, 20, 30],
        2: [20, 30, 40],
        3: [30, 40, 50],
    }

    segmentation_image = np.ones(shape=(10, 10, 3)) * void_color
    segmentation_image[:3, :3] = indices_to_colors_map[1]
    segmentation_image[:3, 7:] = indices_to_colors_map[2]
    segmentation_image[8:, 6:] = indices_to_colors_map[3]

    expected = np.zeros(shape=(10, 10))
    expected[:3, :3] = 1
    expected[:3, 7:] = 2
    expected[8:, 6:] = 3

    actual = net.processing.get_sparse_segmentation_labels_image(segmentation_image, indices_to_colors_map)

    assert np.all(expected == actual)


def test_get_dense_segmentation_labels_image():
    """
    Test get_dense_segmentation_labels_image
    """

    segmentation_image = np.array([
        [4, 1, 2, 2],
        [2, 2, 3, 3]
    ], dtype=np.uint8)

    indices_to_colors_map = {
        1: [10, 10, 10],
        2: [20, 20, 20],
        3: [30, 30, 30],
        4: [40, 40, 40],
    }

    expected = np.array([
        [[40, 40, 40], [10, 10, 10], [20, 20, 20], [20, 20, 20]],
        [[20, 20, 20], [20, 20, 20], [30, 30, 30], [30, 30, 30]],
    ], dtype=np.uint8)

    actual = net.processing.get_dense_segmentation_labels_image(
        segmentation_image, indices_to_colors_map)

    assert np.all(expected == actual)


def test_are_any_target_colors_present_in_image_when_target_colors_present_in_image():
    """
    Test net.processing.are_any_target_colors_present_in_image when target colors are present in image
    """

    image = np.array([
        [1, 2, 3], [2, 3, 4], [3, 4, 5],
        [5, 2, 3], [7, 3, 4], [8, 4, 5],
    ]).reshape((2, 3, 3))

    colors = [(2, 3, 4), (7, 7, 7)]

    assert net.processing.are_any_target_colors_present_in_image(image, colors) is True


def test_are_any_target_colors_present_in_image_when_target_colors_are_nopresent_in_image():
    """
    Test net.processing.are_any_target_colors_present_in_image are not present in image
    """

    image = np.array([
        [1, 2, 3], [2, 3, 4], [3, 4, 5],
        [5, 2, 3], [7, 3, 4], [8, 4, 5],
    ]).reshape((2, 3, 3))

    colors = [(2, 3, 6), (7, 7, 7)]

    assert net.processing.are_any_target_colors_present_in_image(image, colors) is False
