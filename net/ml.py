"""
Module with machine learning code
"""

import tensorflow as tf


class DeepLabV3PlusBuilder:
    """
    Helper class for building DeepLabV3 model from
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" paper
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.activation = tf.nn.swish

    def get_model(self, categories_count: int) -> tf.keras.Model:
        """
        Model builder functions

        Args:
            categories_count (int): number of categories to predict, including background

        Returns:
            tf.keras.Model: DeepLabV3 model
        """

        input_op = tf.keras.layers.Input(shape=(None, None, 3))

        features = self._get_features_extractor(input_op=input_op)
        decoded_features = self._get_decoder(input_op=features)

        predictions_op = tf.keras.layers.Conv2D(
            filters=categories_count, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.softmax
        )(decoded_features)

        model = tf.keras.Model(
            inputs=input_op,
            outputs=[predictions_op]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
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

        for filters in [128, 256, 512]:

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

        x = tf.keras.layers.SeparableConv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.SeparableConv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.SeparableConv2D(
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

        # for _ in range(16):
        for _ in range(4):

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

            x = tf.keras.layers.SeparableConv2D(
                filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
            )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        return x + input_op

    def _get_exit_flow_segment(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get exit flow segment part of the network

        Args:
            input_op (tf.Tensor): input op into the segment

        Returns:
            tf.Tensor: output op
        """

        skip_connection = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(1, 1), strides=(2, 2), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.SeparableConv2D(
            filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.keras.layers.SeparableConv2D(
            filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=self.activation
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = x + skip_connection

        x = tf.keras.layers.SeparableConv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        return x

    def _get_features_extractor(self, input_op) -> tf.Tensor:
        """
        Get feature extractor

        Args:
            input_op (tf.Tensor): input tensor

        Returns:
            tf.Tensor: output op
        """

        x = self._get_entry_flow_segment(input_op=input_op)
        x = self._get_middle_flow_segment(input_op=x)
        x = self._get_exit_flow_segment(input_op=x)
        return x

    def _get_decoder(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get decoder

        Args:
            input_op (tf.Tensor): input tensor

        Returns:
            tf.Tensor: output op
        """

        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", activation=self.activation
        )(input_op)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=self.activation
        )(x)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        return x


class NewDeepLabBuilder:
    """
    Helper class for building DeepLabV3 model from
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" paper
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.activation = tf.nn.swish

    def get_model(self, categories_count: int) -> tf.keras.Model:
        """
        Model builder functions

        Args:
            categories_count (int): number of categories to predict, including background

        Returns:
            tf.keras.Model: DeepLabV3 model
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(None, None, 3)
        )

        input_op = base_model.input
        x = [layer for layer in base_model.layers if layer.name == "conv4_block6_out"][0].output

        decoded_features = self._get_decoder(input_op=x, categories_count=categories_count)

        predictions_op = tf.keras.layers.Conv2D(
            filters=categories_count, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=tf.nn.softmax
        )(decoded_features)

        model = tf.keras.Model(
            inputs=input_op,
            outputs=[predictions_op]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _get_decoder(self, input_op: tf.Tensor, categories_count: int) -> tf.Tensor:
        """
        Get decoder

        Args:
            input_op (tf.Tensor): input tensor
            categories_count (int): number of categories to predict

        Returns:
            tf.Tensor: output op
        """

        x = self._get_atrous_spacial_pooling_pyramid_output(input_op)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        x = tf.keras.layers.SeparableConv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=self.activation)(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=categories_count, kernel_size=(1, 1), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=None)(x)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        return x

    def _get_atrous_spacial_pooling_pyramid_output(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get atrous spacial pooling pyramid output

        Args:
            input_op (tf.Tensor): input tensor

        Returns:
            tf.Tensor: output op
        """

        dilation_rates = [6, 12, 18, 24]
        outputs = []

        for dilation_rate in dilation_rates:

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                dilation_rate=(dilation_rate, dilation_rate), activation=self.activation)(input_op)

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                dilation_rate=(1, 1), activation=self.activation)(x)

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                dilation_rate=(1, 1), activation=self.activation)(x)

            outputs.append(x)

        return tf.concat(outputs, axis=-1)
