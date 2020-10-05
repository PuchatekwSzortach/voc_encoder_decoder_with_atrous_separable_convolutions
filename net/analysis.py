"""
Module with analysis functionality
"""

import collections
import typing

import numpy as np
import tensorflow as tf
import tqdm

import net.data


class ModelAnalyzer:
    """
    Class for performing model analysis
    """

    def __init__(
            self,
            prediction_model: tf.keras.Model,
            data_loader: net.data.TrainingDataLoader,
            categories: typing.List[str]) -> None:
        """
        Constructor

        Args:
            prediction_model (tf.keras.Model): prediction model
            data_loader (net.data.TrainingDataLoader): data loader instance
            categories (typing.List[str]): list of categories
        """

        self.prediction_model = prediction_model
        self.data_loader = data_loader
        self.categories = categories

    def analyze_intersection_over_union(self):
        """
        Analyze intersection over union
        """

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self.data_loader),
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([None, None, None, 3]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, None]))
        ).prefetch(10)

        iterator = iter(dataset)

        categories_intersections_counts = collections.defaultdict(int)
        categories_unions_counts = collections.defaultdict(int)

        for _ in tqdm.tqdm(range(len(self.data_loader))):

            batch_categories_intersections_counts, batch_categories_unions_counts = \
                self._get_iou_results_for_single_batch(iterator)

            for category in self.categories:

                categories_intersections_counts[category] += batch_categories_intersections_counts[category]
                categories_unions_counts[category] += batch_categories_unions_counts[category]

        print("Intersection over union across categories")
        for category in self.categories:

            print(f"{category}: {categories_intersections_counts[category] / categories_unions_counts[category]:.4f}")

    def _get_iou_results_for_single_batch(self, iterator):
        """
        Yield one batch from iterator and compute intersections and unions counts on that batch

        Args:
            iterator (Iterable): iterator yielding (images, ground truth segmentations, masks) batches
        Returns:
            [Tuple]: two dictionaries, first with categories: intersections pixels count and second with
            categories: union pixels count data
        """

        images, ground_truth_segmentations, masks = next(iterator)

        predictions = self.prediction_model.predict(images)

        categories_intersections_counts = collections.defaultdict(int)
        categories_unions_counts = collections.defaultdict(int)

        for ground_truth_segmentation, mask, prediction in zip(ground_truth_segmentations, masks, predictions):

            for index, category in enumerate(self.categories):

                sparse_ground_truth_segmentation = ground_truth_segmentation * mask
                sparse_prediction = np.argmax(prediction, axis=-1)

                categories_intersections_counts[category] += \
                    np.sum(np.logical_and(sparse_ground_truth_segmentation == index, sparse_prediction == index))

                categories_unions_counts[category] += \
                    np.sum(np.logical_or(sparse_ground_truth_segmentation == index, sparse_prediction == index))

        return categories_intersections_counts, categories_unions_counts
