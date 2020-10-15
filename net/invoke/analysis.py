"""
Module with analysis tasks
"""

import invoke


@invoke.task
def analyze_image_sizes(_context, config_path):
    """
    Analyze dataset image sizes

    Args:
        _context (invoke.Context): context instance
        config_path (str): path configuration file
    """

    import glob
    import os

    import PIL.Image
    import tqdm

    import net.utilities

    config = net.utilities.read_yaml(config_path)

    paths = \
        glob.glob(os.path.join(config["voc_data_images_directory"], "**/*.jpg"), recursive=True) + \
        glob.glob(os.path.join(config["hariharan_data_images_directory"], "**/*.jpg"), recursive=True)

    sizes = [PIL.Image.open(path).size for path in tqdm.tqdm(paths)]

    widths, heights = zip(*sizes)

    print(f"max width: {max(widths)}")
    print(f"max height: {max(heights)}")


@invoke.task
def analyze_model(_context, config_path):
    """
    Analyze model's performance

    Args:
        _context (invoke.Context): context instance
        config_path (str): path configuration file
    """
    import tensorflow as tf

    import net.analysis
    import net.data
    import net.logging
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

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

    prediction_model = tf.keras.models.load_model(filepath=config["current_model_directory"])

    net.analysis.ModelAnalyzer(
        mlflow_tracking_uri=config["mlflow_tracking_uri"],
        prediction_model=prediction_model,
        data_loader=validation_samples_data_loader,
        categories=config["categories"]
    ).analyze_intersection_over_union()
