#-------------- Base Image -------------------
FROM jupyter/minimal-notebook:python-3.11 as BASE

ARG CODE_DIR=/tmp/code
ARG POETRY_VERSION=1.6.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    POETRY_VERSION=$POETRY_VERSION \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    CODE_DIR=$CODE_DIR

ENV PATH="${POETRY_HOME}/bin:$PATH"

USER root

RUN curl -sSL https://install.python-poetry.org | python -

USER ${NB_UID}

WORKDIR $CODE_DIR

COPY --chown=${NB_UID}:${NB_GID} poetry.lock pyproject.toml ./

RUN poetry install --no-interaction --no-ansi --no-root --only main
RUN poetry install --no-interaction --no-ansi --no-root --with add1
RUN poetry install --no-interaction --no-ansi --no-root --with add2
RUN poetry install --no-interaction --no-ansi --no-root --with control
RUN poetry install --no-interaction --no-ansi --no-root --with offline

COPY --chown=${NB_UID}:${NB_GID} src/ src/
COPY --chown=${NB_UID}:${NB_GID} README.md .

RUN poetry build


#-------------- Main Image -------------------
FROM jupyter/minimal-notebook:python-3.11 as MAIN

ARG CODE_DIR=/tmp/code

ENV DEBIAN_FRONTEND=noninteractive\
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    CODE_DIR=$CODE_DIR

ENV PATH="${CODE_DIR}/.venv/bin:$PATH"

USER root

# pandoc needed for docs, see https://nbsphinx.readthedocs.io/en/0.7.1/installation.html?highlight=pandoc#pandoc
# gh-pages action uses rsync
# opengl and ffmpeg needed for rendering envs. These packages are needed for torcs and mujoco.
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    pandoc git-lfs rsync ffmpeg x11-xserver-utils patchelf libglew-dev  \
    make g++ gdb libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev  \
    libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev libosmesa6-dev \
    libxrender-dev libxrandr-dev libpng-dev libxxf86vm-dev libvorbis-dev xautomation\
    && rm -rf /var/lib/apt/lists/* \

RUN touch ~/.Xauthority

USER ${NB_UID}

WORKDIR ${CODE_DIR}

# Copy virtual environment from base image
COPY --from=BASE ${CODE_DIR}/.venv ${CODE_DIR}/.venv
# Copy built package from base image
COPY --from=BASE ${CODE_DIR}/dist ${CODE_DIR}/dist

# This goes directly into main jupyter, not poetry env
COPY --chown=${NB_UID}:${NB_GID} build_scripts ./build_scripts
RUN bash build_scripts/install_presentation_requirements.sh


# Install Mujoco
WORKDIR ${CODE_DIR}
RUN mkdir -p ${CODE_DIR}/.mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
RUN tar -xf mujoco.tar.gz -C ${CODE_DIR}/.mujoco
RUN rm mujoco.tar.gz
RUN chmod -R a+rx ${CODE_DIR}/.mujoco


# Start of HACK: the home directory is overwritten by a mount when a jhub server is started off this image
# Thus, we create a jovyan-owned directory to which we copy the code and then move it to the home dir as part
# of the entrypoint
COPY --chown=${NB_UID}:${NB_GID} entrypoint.sh $CODE_DIR
RUN chmod +x "${CODE_DIR}/"entrypoint.sh

# Unfortunately, we cannot use ${CODE_DIR} in the ENTRYPOINT directive, so we have to hardcode it
# Keep in sync with the value of CODE_DIR above
ENTRYPOINT ["/tmp/code/entrypoint.sh"]

# End of HACK

WORKDIR "${HOME}"

COPY --chown=${NB_UID}:${NB_GID} . $CODE_DIR

# Move to the code dir to install dependencies as the CODE_DIR contains the
# complete code base, including the poetry.lock file
WORKDIR $CODE_DIR

RUN pip install --no-cache-dir dist/*.whl

RUN ipython kernel install --name "tfl-training-rl" --user

RUN find notebooks -name '*.ipynb' -exec jupyter trust {} \;

