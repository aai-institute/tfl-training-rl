import os


def get_offline_rl_abs_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_custom_envs_abs_path():
    return os.path.join(get_offline_rl_abs_path(), "custom_envs")


def get_gym_torcs_abs_path():
    return os.path.join(get_custom_envs_abs_path(), "gym_torcs")

