"""
Module with docker related commands
"""

import invoke


@invoke.task
def run(context):
    """
    Run docker container for the app

    :param context: invoke.Context instance
    """

    import os

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else ""
    }

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app:delegated "
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
def build_app_base_container(context, tag):
    """
    Build app base container

    :param context: invoke.Context instance
    :param context: tag: str, tag for the image
    """

    command = (
        "docker build "
        f"--tag puchatek_w_szortach/voc_encoder_decoder_with_atrous_separable_convolutions_base:{tag} "
        "-f ./docker/app_base.Dockerfile ."
    )

    context.run(command, echo=True)
