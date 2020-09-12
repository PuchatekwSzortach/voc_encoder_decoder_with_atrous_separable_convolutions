FROM tensorflow/tensorflow:2.2.0-gpu

# Install a few useful libs and apps
RUN apt update && apt install -y wget vim git

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Add user for the container
RUN useradd -u 1010 -ms /bin/bash app_user

# Setup bashrc for app user
COPY ./docker/bashrc /home/app_user/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# Select user container should be run with
USER app_user

# Set up working directory
WORKDIR /app
