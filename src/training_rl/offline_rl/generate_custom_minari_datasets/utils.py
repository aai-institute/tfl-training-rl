from training_rl.offline_rl.custom_envs.utils import Grid2DInitialConfig


def generate_compatible_minari_dataset_name(env_name: str, data_set_name: str, version: str):
    full_data_set_name = env_name + "-" + data_set_name + "-" + version
    return full_data_set_name


def get_dataset_name_2d_grid(env_config: Grid2DInitialConfig) -> str:
    obstacle_name = env_config.obstacles
    if obstacle_name is not None:
        obstacle_name = f"_{obstacle_name.name}"
    initial_state = env_config.initial_state
    if initial_state is not None:
        initial_state = f"_start_{initial_state[0]}_{initial_state[1]}"
    target_state = env_config.target_state
    if target_state is not None:
        target_state = f"_target_{target_state[0]}_{target_state[1]}"

    return f"data{obstacle_name}{initial_state}{target_state}"
