# Base image
FROM mambaorg/micromamba:0.25.1
USER root

# Set working directory
WORKDIR /ember

RUN apt-get update && apt-get install -y sudo
# Create a new user 'myuser' and add it to the 'sudo' group

# Install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements_conda.txt /ember/
RUN micromamba install -y -n base --channel conda-forge python=3.6 --file requirements_conda.txt && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy all files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /ember

# Install EMBER
RUN python setup.py install


# RUN wget https://ember.elastic.co/ember_dataset_2018_2.tar.bz2