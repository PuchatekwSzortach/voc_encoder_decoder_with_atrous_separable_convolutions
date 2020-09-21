"""
Module with data related code
"""

import os
import typing

import cv2
import numpy as np

import net.processing


class VOCSamplesDataLoader:
    """
    Class for loading VOC samples
    """

    def __init__(
        self, images_directory: str, segmentations_directory: str, data_set_path, batch_size: int,
        shuffle: bool
            ) -> None:
        """
        Constructor

        Args:
            images_directory (str): path to directory with images
            segmentations_directory (str): path to directory with segmentations
            data_set_path (str): path to file with list of images for dataset. Defined w.r.t. data_directory
            batch_size (int): batch size
            shuffle (bool): indicates if samples should be shuffled
        """

        self.samples_paths = self._get_samples_paths(
            images_directory=images_directory,
            segmentations_directory=segmentations_directory,
            data_set_path=data_set_path
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_samples_paths(
        self, images_directory: str, segmentations_directory: str, data_set_path
            ) -> typing.List[typing.Tuple[str, str]]:
        """
        Get a list of (image_path, segmentation_path) for samples identified by
        data directory and data set path

        Args:
            images_directory (str): path to directory with images
            segmentations_directory (str): path to directory with segmentations
            data_set_path (str): path to file with list of images for dataset. Defined w.r.t. data_directory
        Returns:
            typing.List[typing.Tuple[str, str]: list of tuples (image_path, segmentation_path)
        """

        samples = []

        with open(data_set_path) as file:

            for line in file.readlines():

                file_stem = line.strip()

                image_path = os.path.join(images_directory, "{}.jpg".format(file_stem))
                segmentation_path = os.path.join(segmentations_directory, "{}.png".format(file_stem))

                samples.append((image_path, segmentation_path))

        return samples

    def __len__(self):

        return len(self.samples_paths) // self.batch_size

    def __iter__(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Iterator, yields tuples (images, segmentations)
        """

        samples_paths_array = np.array(self.samples_paths)

        while True:

            if self.shuffle is True:
                np.random.shuffle(samples_paths_array)

            start_index = 0

            while start_index < len(samples_paths_array):

                samples_paths_batch = samples_paths_array[start_index: start_index + self.batch_size]

                samples_batch = self._get_samples_batch(
                    samples_paths=samples_paths_batch
                )

                yield samples_batch

                start_index += self.batch_size

    def _get_samples_batch(
        self,
        samples_paths: typing.List[typing.Tuple[str, str]]
            ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Given batch of samples paths, return batch of samples

        Args:
            samples_paths (typing.List[typing.Tuple[str, str]]): list of samples paths tuples, each
            tuple contains image path and segmentation path
        """

        images = []
        segmentations = []

        for image_path, segmentation_path in samples_paths:

            images.append(
                cv2.imread(image_path)
            )

            segmentations.append(
                cv2.imread(segmentation_path)
            )

        return np.array(images), np.array(segmentations)


class TrainingDataLoader:
    """
    Data loader that yields batches (images, segmentations) suitable for training segmentation model
    """

    def __init__(self, samples_data_loader: VOCSamplesDataLoader, use_training_mode: bool) -> None:
        """
        Constructor

        Args:
            samples_data_loader (VOCSamplesDataLoader): samples data loader
            use_training_mode (bool): specifies if training or validation data mode should be used.
        """

        self.samples_data_loader = samples_data_loader
        self.use_training_model = use_training_mode

    def __len__(self):

        return len(self.samples_data_loader)

    def __iter__(self):

        iterator = iter(self.samples_data_loader)

        while True:

            images, segmentations = next(iterator)

            yield self._process_batch(images, segmentations)

    def _process_batch(self, images: np.ndarray, segmentations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Process batch into format suitable for training

        Args:
            images (np.ndarray): batch of images
            segmentations (np.ndarray): batch of segmentations
        """

        # Pad images and segmentations to a fixed size
        processed_images = []
        processed_segmentations = []

        for image, segmentation in zip(images, segmentations):

            processed_images.append(
                net.processing.pad_to_size(
                    image=image, size=512, color=(0, 0, 0)
                )
            )

            processed_segmentations.append(
                net.processing.pad_to_size(
                    image=segmentation,
                    size=512,
                    color=(0, 0, 0)
                )
            )

        return np.array(processed_images), np.array(processed_segmentations)
