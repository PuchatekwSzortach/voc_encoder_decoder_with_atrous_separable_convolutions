"""
Module with machine learning code
"""

import tensorflow as tf


class DeepLabV3Builder:
    """
    Helper class for building DeepLabV3 model from
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" paper
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.activation = tf.nn.swish

    def get_model(self) -> tf.keras.Model:
        """
        Model builder functions

        Returns:
            tf.keras.Model: DeepLabV3 model
        """

        input_op = tf.keras.layers.Input(shape=(None, None, 3))

        x = self._get_entry_flow_segment(input_op=input_op)

        model = tf.keras.Model(
            inputs=input_op,
            outputs=[x]
        )

        return model

    def _get_entry_flow_segment(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get entry flow segment part of the network

        Args:
            input_op (tf.Tensor): input op into the segment

        Returns:
            tf.Tensor: output op
        """

        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        for filters in [128, 256, 728]:

            x = self._get_entry_flow_block(input_op=x, filters=filters)

        return x

    def _get_entry_flow_block(self, input_op: tf.Tensor, filters: int) -> tf.Tensor:
        """
        Get a single entry flow block

        Args:
            input_op (tf.Tensor): input op into the block
            filters (int): number of filters convolutions layers in the block should use

        Returns:
            tf.Tensor: output op
        """

        skip_connection = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=(2, 2), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.BatchNormalization()(x)

        return x + skip_connection

    def _get_middle_flow_segment(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get middle flow segment part of the network

        Args:
            input_op (tf.Tensor): input op into the segment

        Returns:
            tf.Tensor: output op
        """

        x = input_op

        for _ in range(16):

            x = self._get_middle_flow_block(input_op=x)

        return x

    def _get_middle_flow_block(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get a single middle flow block

        Args:
            input_op (tf.Tensor): input op into the block

        Returns:
            tf.Tensor: output op
        """

        x = input_op

        for _ in range(3):

            x = tf.keras.layers.Conv2D(
                filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
            )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        return x + input_op
