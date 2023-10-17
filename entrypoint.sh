#!/bin/bash

shopt -s dotglob

ROOT_DIR="${HOME}"/tfl-training-rl

if [ ! -d  "${ROOT_DIR}" ]; then
  echo "Code not found in ${ROOT_DIR}, copying it during entrypoint. With jupyterhub this should happen only once"
  mkdir "${ROOT_DIR}"
  cp -rf "${CODE_DIR}"/* "${ROOT_DIR}/"
fi

cd "${ROOT_DIR}" || exit

# use poetry for package mgmt.
poetry run jupyter trust notebooks/*.ipynb
poetry run python -m ipykernel install --user --name intro-bayesian-ml


# original entrypoint, see https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile#L150
# need -s option for tini to work properly when started not as PID 1
tini -s -g -- "$@"
