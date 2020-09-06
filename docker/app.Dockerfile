# Dockerfile for app container
FROM puchatek_w_szortach/voc_encoder_decoder_with_atrous_separable_convolutions_base:0.1.0

# Update python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

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

# Copy app code
COPY . /app
