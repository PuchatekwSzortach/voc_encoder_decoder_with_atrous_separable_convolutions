"""
Module with logging utilities
"""

import math

import logging
import typing

import numpy as np
import tensorflow as tf
import vlogging

import net.processing


def log_predictions(
        logger: logging.Logger, prediction_model: tf.keras.Model,
        images: typing.List[np.ndarray], ground_truth_segmentations: typing.List[np.ndarray],
        categories: typing.List[str]):
    """
    Log a batch of predictions, along with input images and ground truth segmentations

    Args:
        logger (logging.Logger): logger instance
        prediction_model (tf.keras.Model): prediction model
        images (typing.List[np.ndarray]): list of images to run predictions on
        ground_truth_segmentations (typing.List[np.ndarray]): list of ground truth segmentations
        categories (typing.List[str]): list of segmentation categories
    """

    images = [net.processing.pad_to_size(
        image=image,
        size=math.ceil(max(image.shape) / 32) * 32,
        color=(0, 0, 0)
    ) for image in images]

    logger.info(
        vlogging.VisualRecord(
            title="images",
            imgs=list(images)
        )
    )

    logger.info(
        vlogging.VisualRecord(
            title="ground truth segmentations",
            imgs=list(ground_truth_segmentations)
        )
    )

    indices_to_colors_map = net.data.get_colors_info(categories_count=len(categories))[0]

    one_hot_encoded_predictions = [prediction_model.predict(np.array([image]))[0] for image in images]

    bgr_predictions = [net.processing.get_dense_segmentation_labels_image(
        segmentation_image=np.argmax(prediction, axis=-1),
        indices_to_colors_map=indices_to_colors_map) for prediction in one_hot_encoded_predictions]

    logger.info(
        vlogging.VisualRecord(
            title="segmentations predictions",
            imgs=bgr_predictions
        )
    )
