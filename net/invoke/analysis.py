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
        glob.glob(os.path.join(config["harikaran_data_images_directory"], "**/*.jpg"), recursive=True)

    sizes = [PIL.Image.open(path).size for path in tqdm.tqdm(paths)]

    widths, heights = zip(*sizes)

    print(f"max width: {max(widths)}")
    print(f"max height: {max(heights)}")
