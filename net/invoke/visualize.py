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
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        batch_size=config["batch_size"]
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, segmentations = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="images",
                imgs=images
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=segmentations
            )
        )
