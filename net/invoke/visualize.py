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

    import tqdm
    import vlogging

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_training_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=True
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="images",
                imgs=list(images)
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=list(segmentations)
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

    import tqdm
    import vlogging

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    samples_data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_training_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=True
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
                imgs=list(images)
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=[10 * image for image in segmentations]
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="masks",
                imgs=[255 * image for image in masks]
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
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_training_samples_list_path"],
        batch_size=config["batch_size"],
        shuffle=True
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    prediction_model = tf.keras.models.load_model(filepath=config["current_model_directory"])

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        net.logging.log_predictions(
            logger=logger,
            prediction_model=prediction_model,
            images=images,
            ground_truth_segmentations=segmentations,
            categories=config["categories"]
        )
