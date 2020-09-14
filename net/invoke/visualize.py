"""
Module with visualization related tasks
"""

import invoke


@invoke.task
def visualize_data(_context: invoke.Context, config_path: str):
    """
    Visualize a few data samples

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    pass
