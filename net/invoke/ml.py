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
    import net.ml
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

    iterator = iter(training_samples_data_loader)
    model = net.ml.DeepLabV3PlusBuilder().get_model()

    for _ in range(4):

        images, _, _ = next(iterator)
        print(f"images shape: {images.shape}")

        predictions = model.predict(images)
        print(f"predictions shape: {predictions.shape}\n")
