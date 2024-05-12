#!/bin/bash

shopt -s dotglob

ROOT_DIR="${HOME}"/tfl-training-rl

if [ ! -d  "${ROOT_DIR}" ]; then
  echo "Code not found in ${ROOT_DIR}, copying it during entrypoint. With jupyterhub this should happen only once"
  mkdir "${ROOT_DIR}"
  cp -rf "${CODE_DIR}"/* "${ROOT_DIR}/"
fi

# Move .mujoco folder to the user's home directory. Should be done in Dockerfile but some issues.
# (likely because of the hack in Dockerfile).
if [ -d "${ROOT_DIR}/.mujoco" ]; then
  mv "${ROOT_DIR}/.mujoco/" "${HOME}/"
fi

cd "${ROOT_DIR}" || exit

#Uninstall mujoco_py and reinstalled. mujoco_py not installed properly in jhub (no issues with local image).
pip uninstall -y mujoco_py
pip install mujoco_py==2.1.2.14

# Install IPython kernel. kernel is not install in jhub (no issues with local image).
ipython kernel install --name "tfl-training-rl" --user


# original entrypoint, see https://github.com/jupyter/docker-stacks/blob/main/images/docker-stacks-foundation/Dockerfile#L131
# need -s option for tini to work properly when started not as PID 1
tini -s -g -- start.sh "$@"
