"""
Module with docker related commands
"""

import invoke


@invoke.task
def run(context, config_path):
    """
    Run docker container for the app

    :param context: invoke.Context instance
    :param config_path: str, path to configuration file
    """

    import os

    import net.utilities

    config = net.utilities.read_yaml(config_path)

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else "",
        "data_directory_on_host": os.path.abspath(config["DATA_DIRECTORY_ON_HOST"]),
        # A bit of sourcery to create data volume that can be shared with docker-compose containers
        "log_data_volume": os.path.basename(os.path.abspath('.') + '_log_data')
    }

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app:delegated "
        "-v {data_directory_on_host}:/data "
        "-v {log_data_volume}:/tmp "
        "puchatek_w_szortach/voc_encoder_decoder_with_atrous_separable_convolutions:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)


@invoke.task
def build_app_container(context):
    """
    Build app container

    :param context: invoke.Context instance
    """

    command = (
        "docker build "
        "--tag puchatek_w_szortach/voc_encoder_decoder_with_atrous_separable_convolutions:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def up(context, config_path):
    """
    Runs docker-compose up, providing it with values for environmental variables

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import net.utilities

    config = net.utilities.read_yaml(config_path)

    # Just load every key for which value is a string
    environmental_variables = {key: value for key, value in config.items() if isinstance(value, str)}
    environmental_variables_string = " ".join([f"{key}={value}" for key, value in environmental_variables.items()])

    context.run(environmental_variables_string + " docker-compose up -d", echo=True, pty=True)
