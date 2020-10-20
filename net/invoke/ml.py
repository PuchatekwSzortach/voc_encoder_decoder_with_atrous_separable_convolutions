"""
Module with machine learning tasks
"""

import invoke


@invoke.task
def train(_context, config_path, load_existing_model=False):
    """
    Task for training a segmentation model

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
        load_existing_model (bool): specifies if existing model should be loaded, instead of training from scratch.
        Defaults to False
    """

    import mlflow
    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment("training")

    with mlflow.start_run(run_name="simple_run"):

        mlflow.tensorflow.autolog(every_n_iter=1)

        trainig_voc_samples_data_loader = net.data.CombinedPASCALDatasetsLoader(
            config=config,
            augmentation_pipeline=net.processing.get_augmentation_pipepline()
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
        ).prefetch(32)

        validation_voc_samples_data_loader = net.data.VOCSamplesDataLoader(
            images_directory=config["voc_data_images_directory"],
            segmentations_directory=config["voc_data_segmentations_directory"],
            data_set_path=config["voc_validation_samples_list_path"],
            batch_size=config["batch_size"],
            shuffle=False
        )

        validation_samples_data_loader = net.data.TrainingDataLoader(
            samples_data_loader=validation_voc_samples_data_loader,
            use_training_mode=False,
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
        ).prefetch(32)

        model = tf.keras.models.load_model(
            filepath=config["current_model_directory"],
            custom_objects={
                "get_temperature_scaled_sparse_softmax": net.ml.get_temperature_scaled_sparse_softmax}
        ) if load_existing_model else \
            net.ml.DeepLabV3PlusBuilder().get_model(categories_count=len(config["categories"]))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=net.ml.get_temperature_scaled_sparse_softmax,
            metrics=['accuracy']
        )

        model.fit(
            x=training_dataset,
            epochs=100,
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
                    patience=8,
                    verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.1,
                    patience=3,
                    verbose=1),
                tf.keras.callbacks.CSVLogger(
                    filename=config["training_metrics_log_path"]
                )
            ]
        )
