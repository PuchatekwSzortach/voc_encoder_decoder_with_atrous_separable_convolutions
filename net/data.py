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

        return images, segmentations


class TrainingDataLoader:
    """
    Data loader that yields batches (images, segmentations) suitable for training segmentation model
    """

    def __init__(
            self, samples_data_loader: VOCSamplesDataLoader,
            use_training_mode: bool,
            size: int,
            categories: typing.List[str]) -> None:
        """
        Constructor

        Args:
            samples_data_loader (VOCSamplesDataLoader): samples data loader
            use_training_mode (bool): specifies if training or validation data mode should be used
            size (int): size to which images should be padded in both directions
            categories (typing.List[str]): list of categories
        """

        self.samples_data_loader = samples_data_loader
        self.use_training_model = use_training_mode
        self.size = size
        self.categories = categories

        self.categories_to_indices_map = {category: index for index, category in enumerate(categories)}
        self.indices_to_colors_map, self.void_color = get_colors_info(len(categories))

    def __len__(self):

        return len(self.samples_data_loader)

    def get_bgr_iterator(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Get iterator that yields (images, segmentations) such that segmentations are in 3 channel images
        in (blue, green, red) order

        Returns:
            (typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]):
            iterator that yields (images, segmentations) batches
        """

        iterator = iter(self.samples_data_loader)

        while True:

            images, segmentations = next(iterator)
            yield self._process_batch(images, segmentations)

    def __iter__(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterator that yields tuples (images, sparse segmentations labels, segmentations masks), where
        sparse segmentations labels are 2D numpy arrays with pixel values corresponding to categories,
        and segmentations mask are 2D binary masks with 1 set to pixels that have represent valid segments,
        and 0 set to pixels with vague segments/pixels to ignore during trainig.

        Yields:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple (images, segmentations, segmentations masks)
        """

        iterator = self.get_bgr_iterator()

        while True:

            images, segmentations = next(iterator)

            sparse_segmentations = [net.processing.get_sparse_segmentation_labels_image(
                segmentation_image=segmentation,
                indices_to_colors_map=self.indices_to_colors_map
            ) for segmentation in segmentations]

            # Masks that are set to 0 in pixels that have void color, and thus should be ignored during training
            masks = [
                np.all(segmentation != self.void_color, axis=-1).astype(np.int32) for segmentation in segmentations]

            yield images.astype(np.float32), np.array(sparse_segmentations, dtype=np.float32), np.array(masks)

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
                    image=image,
                    size=self.size,
                    color=self.indices_to_colors_map[self.categories_to_indices_map["background"]]
                )
            )

            processed_segmentations.append(
                net.processing.pad_to_size(
                    image=segmentation,
                    size=self.size,
                    color=self.indices_to_colors_map[self.categories_to_indices_map["background"]]
                )
            )

        return np.array(processed_images), np.array(processed_segmentations)


def get_colors_info(categories_count):
    """
    Get ids to colors dictionary and void color.
    Ids to colors dictionary maps gives colors used in VOC dataset for a given category id.
    Void color represents ambiguous regions in segmentations.
    All colors are returned in BGR order.
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :param categories_count: number of categories - includes background, but doesn't include void
    :return: map, tuple
    """

    colors_count = 256

    def bitget(byte_value, idx):
        """
        Check if bit at given byte index is set
        :param byte_value: byte
        :param idx: index
        :return: bool
        """
        return (byte_value & (1 << idx)) != 0

    colors_matrix = np.zeros(shape=(colors_count, 3), dtype=np.int)

    for color_index in range(colors_count):

        red = green = blue = 0
        color = color_index

        for j in range(8):

            red = red | (bitget(color, 0) << 7 - j)
            green = green | (bitget(color, 1) << 7 - j)
            blue = blue | (bitget(color, 2) << 7 - j)
            color = color >> 3

        # Writing colors in BGR order, since our image reading and logging routines use it
        colors_matrix[color_index] = blue, green, red

    indices_to_colors_map = {color_index: tuple(colors_matrix[color_index]) for color_index in range(categories_count)}
    return indices_to_colors_map, tuple(colors_matrix[-1])
