"""
Module with visualization related tasks
"""

import invoke


@invoke.task
def visualize_data(_context, config_path):
    """
    Visualize a few data samples

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import cv2
    import tqdm
    import vlogging

    import net.data
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.CombinedPASCALDatasetsLoader(
        voc_data_directory=config["voc_data_directory"],
        hariharan_data_directory=config["hariharan_data_directory"],
        categories_count=len(config["categories"]),
        batch_size=config["batch_size"],
        augmentation_pipeline=net.processing.get_augmentation_pipepline()
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="images",
                imgs=[cv2.pyrDown(image) for image in images]
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=[cv2.pyrDown(image) for image in segmentations]
            )
        )


@invoke.task
def visualize_training_samples(_context, config_path):
    """
    Visualize a few training samples

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import cv2
    import tqdm
    import vlogging

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    samples_data_loader = net.data.CombinedPASCALDatasetsLoader(
        voc_data_directory=config["voc_data_directory"],
        hariharan_data_directory=config["hariharan_data_directory"],
        categories_count=len(config["categories"]),
        batch_size=config["batch_size"],
        augmentation_pipeline=net.processing.get_augmentation_pipepline()
    )

    training_samples_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=samples_data_loader,
        use_training_mode=True,
        size=config["training_image_dimension"],
        categories=config["categories"]
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    iterator = iter(training_samples_data_loader)

    for _ in tqdm.tqdm(range(10)):

        images, segmentations, masks = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="images",
                imgs=[cv2.pyrDown(image) for image in images]
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=[10 * cv2.pyrDown(image) for image in segmentations]
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="masks",
                imgs=[255 * cv2.pyrDown(image) for image in masks]
            )
        )


@invoke.task
def visualize_predictions(_context, config_path):
    """
    Visualize predictions on a few images

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tensorflow as tf
    import tqdm

    import net.data
    import net.logging
    import net.ml
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_validation_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=True
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    prediction_model = tf.keras.models.load_model(
        filepath=config["current_model_directory"],
        custom_objects={
            "get_temperature_scaled_sparse_softmax": net.ml.get_temperature_scaled_sparse_softmax
        }
    )

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        net.logging.log_predictions(
            logger=logger,
            prediction_model=prediction_model,
            images=images,
            ground_truth_segmentations=segmentations,
            categories=config["categories"],
            target_size=config["training_image_dimension"]
        )
