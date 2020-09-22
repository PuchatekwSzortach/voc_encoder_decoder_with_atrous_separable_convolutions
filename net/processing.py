"""
Module with processing functions
"""

import typing

import numpy as np


def pad_to_size(image: np.ndarray, size: int, color: typing.Tuple[int, int, int]) -> np.ndarray:
    """
    Given an image center-pad with selecter color to given size in both dimensions.

    Args:
        image (np.ndarray): 3D numpy array
        size (int): size to which image should be padded in both directions
        color (typing.Tuple[int, int, int]): color that should be used for padding

    Returns:
        np.array: padded images
    """

    # Compute paddings
    total_vertical_padding = size - image.shape[0]

    upper_padding = total_vertical_padding // 2
    lower_padding = total_vertical_padding - upper_padding

    total_horizontal_padding = size - image.shape[1]

    left_padding = total_horizontal_padding // 2
    right_padding = total_horizontal_padding - left_padding

    # Create canvas with desired shape and background image, paste image on top of it
    canvas = np.ones(shape=(size, size, 3)) * color
    canvas[upper_padding:-lower_padding, left_padding:-right_padding, :] = image

    # Return canvas
    return canvas
