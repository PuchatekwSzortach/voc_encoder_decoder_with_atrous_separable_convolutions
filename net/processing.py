"""
Module with processing functions
"""

import typing

import cv2
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
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Affine(shear={"x": (-20, 20)}),
                imgaug.augmenters.Affine(shear={"y": (-20, 20)}),
            ])
    ])


def are_any_target_colors_present_in_image(
        image: np.ndarray, colors: typing.List[typing.Tuple[int, int, int]]) -> bool:
    """
    Check if image contains any of target colors

    Args:
        image (np.ndarray): 3D array that should be examined for target colors
        colors (typing.List[typing.Tuple[int, int, int]]): list of colors to search for

    Returns:
        bool: True if any of target colors is found in image, False otherwise
    """

    for color in colors:

        # inner np.all checks that for given pixel in image all three components of a color are correct,
        # then outer any checks that there was any pixel with given color
        if any(np.all(image.reshape(-1, 3) == color, axis=-1)) is True:
            return True

    # No target color found in image
    return False


def get_segmentation_overlay(
        image: np.ndarray, segmentation: np.ndarray, background_color: typing.Tuple[int, int, int]) -> np.ndarray:
    """
    Get image with segmentation overlaid over it.

    Args:
        image (np.ndarray): 3-channel image
        segmentation (np.ndarray): 3-channel segmentation
        background_color (typing.Tuple[int, int, int]): background color, segmentation pixels that match
        background color won't be overlaid over image

    Returns:
        np.ndarray: Image with segmentation overlaid over it
    """

    blended_image = cv2.addWeighted(image, 0.2, segmentation, 0.8, 0)
    overlay = image.copy()

    mask = np.logical_not(np.all(segmentation == background_color, axis=-1))
    overlay[mask] = blended_image[mask]

    return overlay
