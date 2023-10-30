# TransferLab Training: Safe and Efficient RL

Welcome to the TransferLab training: Safe and Efficient RL.
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
For running without docker, create a conda environment (with python 3.9),
e.g., with `conda create -n training_rl python=3.9`.
Then, install the dependencies and the package with
    ```shell
    bash build_scripts/install_presentation_requirements.sh
    pip install -e .
    ```

3. If you want to use docker instead, you can build the image locally.
First, set the variable `PARTICIPANT_BUCKET_READ_SECRET` to the secret found in
`config.yaml`, and then build the image with
    ```shell
    docker build --build-arg PARTICIPANT_BUCKET_READ_SECRET=$PARTICIPANT_BUCKET_READ_SECRET -t training_rl .
    ```
    You can then start the container e.g., with
    ```shell
    docker run -it -p 8888:8888 training_rl jupyter notebook
    ```
4. The data will be downloaded on the fly when you run the notebooks.
5. Finally, for creating source code documentation, you can run
    ```shell
    bash build_scripts/build_docs.sh
    ```
    and then open `docs/build/html/index.html` in your browser.
    This will also rebuild the jupyter-book based notebook documentation
    that was originally found in the `html` directory.

6. In case you experience some issues with the rendering when using docker
make sure to add the docker user to xhost. So run on your local machine: 

xhost +SI:localuser:docker_user

and run docker like: 

docker run -p 8888:8888 -it --env DISPLAY=$DISPLAY --net=host --privileged --volume /tmp/.X11-unix:/tmp/.X11-unix  training_rl bash



Note that there is some non-trivial logic in the entrypoint that may collide
with mounting volumes to paths directly inside 
`/home/jovyan/training_rl`. If you want to do that, 
the easiest way is to override the entrypoint or to mount somewhere else
and create a symbolic link. For details on that see the `Dockerfile` and
`entrypoint.sh`.