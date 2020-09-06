# Dockerfile for base container on which app container is based
# Responsible for installing software and other expensive operations that don't have to be changed often
FROM tensorflow/tensorflow:2.2.0-gpu

# Install a few necessary libs and apps
RUN apt update && apt install -y wget vim git

# Add user for the container
RUN useradd -u 1010 -ms /bin/bash app_user

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
