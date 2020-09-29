"""
Module with machine learning tasks
"""

import invoke


@invoke.task
def train(_context, config_path):
    """
    Task for training a segmentation model

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    trainig_voc_samples_data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_training_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=True
    )

    training_samples_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=trainig_voc_samples_data_loader,
        use_training_mode=True,
        size=config["training_image_dimension"],
        categories=config["categories"]
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(training_samples_data_loader),
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"], 3]),
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"]]),
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"]]))
    ).prefetch(10)

    validation_voc_samples_data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_validation_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=False
    )

    validation_samples_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=validation_voc_samples_data_loader,
        use_training_mode=True,
        size=config["training_image_dimension"],
        categories=config["categories"]
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_samples_data_loader),
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"], 3]),
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"]]),
            tf.TensorShape([None, config["training_image_dimension"], config["training_image_dimension"]]))
    ).prefetch(10)

    model = net.ml.DeepLabV3PlusBuilder().get_model(categories_count=len(config["categories"]))

    model.fit(
        x=training_dataset,
        epochs=10,
        steps_per_epoch=len(training_samples_data_loader),
        validation_data=validation_dataset,
        validation_steps=len(validation_samples_data_loader),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config["current_model_directory"],
                save_best_only=True,
                save_weights_only=False,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.1,
                patience=6,
                verbose=1),
            tf.keras.callbacks.CSVLogger(
                filename=config["training_metrics_log_path"]
            )
        ]
    )
