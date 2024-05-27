FROM spark:3.5.1

# Define arguments
ARG USER_NAME
ARG USER_UID
ARG USER_GID

# Configure environment
ENV SHELL=/bin/bash
ENV USER_HOME=/home/$USER_NAME
ENV PROJECT_HOME=$USER_HOME/project

# Set root user
USER root

# Create non-root user
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME

# Force color prompt
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' $USER_HOME/.bashrc

# Add Git official PPA
RUN apt update \
    && apt install -yq --no-install-recommends \
        software-properties-common \
    && apt-add-repository ppa:git-core/ppa

# Install useful OS packages
RUN apt update \
    && apt install -yq --no-install-recommends \
        openssh-client \
        git \
        htop \
        vim \
        tmux \
        python3-tk \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install useful Python packages
RUN pip install --no-cache-dir \
    pipenv==2022.4.8

# Set working directory
WORKDIR $PROJECT_HOME
COPY Pipfile $PROJECT_HOME
RUN chown -R $USER_UID:$USER_GID $PROJECT_HOME

# Set non-root user
USER $USER_NAME

# Install project dependencies
RUN pipenv install --dev && pipenv --clear
