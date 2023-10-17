FROM jupyter/minimal-notebook:python-3.11

ENV POETRY_VERSION=1.6.1

USER root
RUN apt-get update && apt-get upgrade -y

# pandoc needed for docs, see https://nbsphinx.readthedocs.io/en/0.7.1/installation.html?highlight=pandoc#pandoc
# gh-pages action uses rsync
# gcc, gfortran and libopenblas-dev are needed for slycot, which in turn is needed by the python-control package
# build-essential required for scikit-build
RUN apt-get -y --no-install-recommends install pandoc git-lfs rsync build-essential gcc gfortran libopenblas-dev

USER ${NB_UID}


# Jhub does not support notebook 7 yet, all hell breaks loose if we don't pin it
RUN pip install "notebook<7"
# This goes directly into main jupyter, not poetry env
COPY --chown=${NB_UID}:${NB_GID} build_scripts ./build_scripts
RUN bash build_scripts/install_presentation_requirements.sh


# Install poetry according to
# https://python-poetry.org/docs/#installing-manually
RUN pip install -U setuptools "poetry==$POETRY_VERSION"

WORKDIR /tmp

# Start of HACK: the home directory is overwritten by a mount when a jhub server is started off this image
# Thus, we create a jovyan-owned directory to which we copy the code and then move it to the home dir as part
# of the entrypoint
ENV CODE_DIR=/tmp/code

RUN mkdir $CODE_DIR

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

RUN poetry config virtualenvs.in-project true
RUN poetry install --no-interaction --no-ansi
