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

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    samples_data_loader = net.data.VOCSamplesDataLoader(
        images_directory=config["voc_data_images_directory"],
        segmentations_directory=config["voc_data_segmentations_directory"],
        data_set_path=config["voc_training_samples_list_path"],
        batch_size=config["batch_size"]
    )

    training_samples_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=samples_data_loader,
        use_training_mode=True
    )

    iterator = iter(training_samples_data_loader)

    for _ in range(4):

        images, segmentations = next(iterator)
        print(images.shape)
        print(segmentations.shape)
        print()
