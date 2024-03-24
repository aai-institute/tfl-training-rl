# TransferLab Training: Control and Safe and Efficient RL

Welcome to the TransferLab trainings: Control, Safe and Efficient RL.
This is the readme for the participants of the training.

## During the training

If you are currently participating in the training, you can find the agenda
in the file `AGENDA.md`. Everything is already set up, so feel free
to follow the trainer's presentation or to explore the notebooks and
source code on your own.

## After the training

You have received this file as part of the training materials.

There are multiple ways of viewing/executing the content. 

1. If you just want to view the rendered notebooks, 
   open `html/index.html` in your browser.
2. If you want to execute the notebooks, you will either need to
   install the dependencies or use docker.
   For running without docker, create a [poetry](https://python-poetry.org/) environment (with python 3.11),
   e.g., with `poetry shell`.

   Then, install the dependencies and the package with

   ```shell
   python poetry_install.py
   bash build_scripts/install_presentation_requirements.sh
   ```

3. If you want to use docker instead,
   you can build the image locally using:
    
   ```shell
   docker build -t tfl-training-rl:local .
   ```

   You can then start the container e.g., with
    
   ```shell
   docker run -it -p 8888:8888 tfl-training-rl:local jupyter notebook --ip=0.0.0.0
   ```

4. Finally, for creating source code documentation, you can run
    
   ```shell
   bash build_scripts/build_docs.sh
   ```

   and then open `docs/build/html/index.html` in your browser.
   This will also rebuild the jupyter-book based notebook documentation
   that was originally found in the `html` directory.

6. In case you experience some issues with the rendering when using docker
   make sure to add the docker user to xhost. So run on your local machine: 

   ```shell
   xhost +SI:localuser:docker_user
   ```

   and run docker using: 

    
   ```shell
   docker run -it --rm --privileged --net=host \
      --env DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix \
      tfl-training-rl:local jupyter notebook --ip=0.0.0.0
   ```

> **Note** There is some non-trivial logic in the entrypoint that may collide
with mounting volumes to paths directly inside 
`/home/jovyan/training_rl`. If you want to do that, 
the easiest way is to override the entrypoint or to mount somewhere else
and create a symbolic link. For details on that see the `Dockerfile` and
`entrypoint.sh`.

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

