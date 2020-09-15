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
        "data_directory_on_host": os.path.abspath(config["data_directory_on_host"])
    }

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app:delegated "
        "-v {data_directory_on_host}:/data "
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
