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
                imgs=images.tolist()
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=segmentations.tolist()
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
        use_training_mode=True
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    iterator = iter(training_samples_data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="images",
                # Unpack np.array into a list of numpy arrays
                imgs=[image for image in images]
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                # Unpack np.array into a list of numpy arrays
                imgs=[segmentation for segmentation in segmentations]
            )
        )
