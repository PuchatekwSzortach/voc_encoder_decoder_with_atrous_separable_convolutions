"""
Module with processing functions
"""

import typing

import imgaug
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
    canvas[upper_padding:size - lower_padding, left_padding:size - right_padding, :] = image

    # Return canvas
    return canvas


def remove_borders(image: np.ndarray, target_size: typing.Tuple[int, int]) -> np.ndarray:
    """
    Remove borders from around the image so the output is of target_size.

    Args:
        image (np.ndarray): input image
        target_size (typing.Tuple[int, int]): tuple (height, width) representing target image size

    Returns:
        np.ndarray: output image
    """

    # Compute paddings
    total_vertical_padding = image.shape[0] - target_size[0]

    upper_padding = total_vertical_padding // 2
    lower_padding = total_vertical_padding - upper_padding

    total_horizontal_padding = image.shape[1] - target_size[1]

    left_padding = total_horizontal_padding // 2
    right_padding = total_horizontal_padding - left_padding

    return image.copy()[
        upper_padding: image.shape[0] - lower_padding, left_padding: image.shape[1] - right_padding, :]


def get_sparse_segmentation_labels_image(
        segmentation_image: np.ndarray,
        indices_to_colors_map: typing.Dict[int, typing.Tuple[int, int, int]]) -> np.ndarray:
    """
    Creates a segmentation labels image that translates segmentation color to index value.
    For each pixel without a reference color provided in indices_to_colors_map value 0 is used.

    Args:
        segmentation_image (np.ndarray): 3 channel (blue, green, red) segmentation image
        indices_to_colors_map (typing.Dict[int, typing.Tuple[int, int, int]]): dictionary mapping categories
        indices to colors

    Returns:
        np.ndarray: 2D numpy array of spare segmentation labels
    """

    segmentation_labels_image = np.zeros(segmentation_image.shape[:2])

    for index, color in indices_to_colors_map.items():

        color_pixels = np.all(segmentation_image == color, axis=2)
        segmentation_labels_image[color_pixels] = index

    return segmentation_labels_image


def get_dense_segmentation_labels_image(
        segmentation_image: np.ndarray,
        indices_to_colors_map: typing.Dict[int, typing.Tuple[int, int, int]]) -> np.ndarray:
    """
    Given sparse encoded segmentations image, convert it to bgr segmentations image

    Args:
        segmentation_image (np.ndarray): 2D array, sparse encoded segmentation image
        indices_to_colors_map (dict): dictionary mapping categories indices to bgr colors

    Returns:
        [np.ndarray]: 3D BGR segmentation image
    """

    bgr_segmentation_image = np.zeros(
        shape=(segmentation_image.shape[0], segmentation_image.shape[1], 3),
        dtype=np.uint8)

    for index, color in indices_to_colors_map.items():

        mask = segmentation_image == index
        bgr_segmentation_image[mask] = color

    return bgr_segmentation_image


def get_augmentation_pipepline() -> imgaug.augmenters.Augmenter:
    """
    Get augmentation pipeline
    """

    return imgaug.augmenters.Sequential([
        imgaug.augmenters.Fliplr(p=0.5),
        imgaug.augmenters.SomeOf(
            n=(0, 3),
            children=[
                imgaug.augmenters.Affine(rotate=(-10, 10)),
                imgaug.augmenters.Affine(scale=(0.5, 1.5))
            ])
    ])
