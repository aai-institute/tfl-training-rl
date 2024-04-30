import os


def load_env_variables():
    current_directory = os.path.dirname(__file__)
    mujoco_directory = os.path.expanduser("~/.mujoco")

    env_variables = os.environ.update(
        {
            #"LD_LIBRARY_PATH": os.path.join(mujoco_directory, "mujoco210", "bin") + ":" + "/usr/lib/x86_64-linux-gnu/nvidia/",
            "LD_LIBRARY_PATH": os.path.join(mujoco_directory, "mujoco210", "bin") + ":" + "/usr/lib/nvidia",
            "MINARI_DATASETS_PATH": os.path.join(current_directory, "data", "offline_data"),
            "TRAINED_POLICY_PATH": os.path.join(current_directory, "data", "trained_models_data"),
            # Add more variables as needed
        }
    )

    return env_variables
