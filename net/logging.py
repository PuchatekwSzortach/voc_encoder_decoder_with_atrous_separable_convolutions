"""
Module with logging utilities
"""

import logging
import typing

import cv2
import numpy as np
import tensorflow as tf
import vlogging

import net.data
import net.processing


def log_predictions(
        logger: logging.Logger, prediction_model: tf.keras.Model,
        images: typing.List[np.ndarray], ground_truth_segmentations: typing.List[np.ndarray],
        categories: typing.List[str], target_size: int):
    """
    Log a batch of predictions, along with input images and ground truth segmentations

    Args:
        logger (logging.Logger): logger instance
        prediction_model (tf.keras.Model): prediction model
        images (typing.List[np.ndarray]): list of images to run predictions on
        ground_truth_segmentations (typing.List[np.ndarray]): list of ground truth segmentations
        categories (typing.List[str]): list of segmentation categories
        target_size (int): common size to which all images should be padded for prediction
    """

    padded_images = np.array([net.processing.pad_to_size(
        image=image,
        size=target_size,
        color=(0, 0, 0)
    ) for image in images])

    logger.info(
        vlogging.VisualRecord(
            title="images",
            imgs=[cv2.pyrDown(image) for image in images]
        )
    )

    logger.info(
        vlogging.VisualRecord(
            title="ground truth segmentations",
            imgs=[cv2.pyrDown(image) for image in ground_truth_segmentations]
        )
    )

    indices_to_colors_map = net.data.get_colors_info(categories_count=len(categories))[0]

    bgr_predictions = [net.processing.get_dense_segmentation_labels_image(
        segmentation_image=np.argmax(prediction, axis=-1),
        indices_to_colors_map=indices_to_colors_map)
        for prediction in prediction_model.predict(padded_images)]

    borderless_bgr_predictions = [net.processing.remove_borders(
        image=prediction,
        target_size=ground_truth_image.shape[:2]
    ) for prediction, ground_truth_image in zip(bgr_predictions, images)]

    logger.info(
        vlogging.VisualRecord(
            title="segmentations predictions",
            imgs=[cv2.pyrDown(image) for image in borderless_bgr_predictions]
        )
    )
