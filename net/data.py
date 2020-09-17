"""
Module with data related code
"""

import os
import typing

import cv2


class VOCSamplesDataLoader:
    """
    Class for loading VOC samples
    """

    def __init__(self, data_directory: str, data_set_path: str, batch_size: int) -> None:
        """
        Constructor

        Args:
            data_directory (str): path to directory with data
            data_set_path (str): path to file with list of images for dataset. Defined w.r.t. data_directory
            batch_size (int): batch size
        """

        self.samples_paths = self._get_samples_paths(
            data_directory=data_directory,
            data_set_path=data_set_path
        )

        self.batch_size = batch_size

    def _get_samples_paths(self, data_directory: str, data_set_path: str) -> typing.List[typing.Tuple[str, str]]:
        """
        Get a list of (image_path, segmentation_path) for samples identified by
        data directory and data set path

        Args:
            data_directory (str): [description]
            data_set_path (str): [description]
        Returns:
            typing.List[typing.Tuple[str, str]: list of tuples (image_path, segmentation_path)
        """

        samples = []

        with open(os.path.join(data_directory, data_set_path)) as file:

            for line in file.readlines():

                file_stem = line.strip()

                image_path = os.path.join(data_directory, "JPEGImages/{}.jpg".format(file_stem))
                segmentation_path = os.path.join(data_directory, "SegmentationClass/{}.png".format(file_stem))

                samples.append((image_path, segmentation_path))

        return samples

    def __len__(self):

        return len(self.samples_paths) // self.batch_size

    def __iter__(self) -> typing.Iterator[typing.Tuple[typing.List, typing.List]]:
        """
        Iterator, yields tuples (images, segmentations)
        """

        while True:

            start_index = 0

            while start_index < len(self.samples_paths):

                samples_paths_batch = self.samples_paths[start_index: start_index + self.batch_size]

                samples_batch = self._get_samples_batch(
                    samples_paths=samples_paths_batch
                )

                yield samples_batch

                start_index += self.batch_size

    def _get_samples_batch(
        self,
        samples_paths: typing.List[typing.Tuple[str, str]]
            ) -> typing.Tuple[typing.List, typing.List]:
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

        return images, segmentations
