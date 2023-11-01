import gymnasium as gym

from training_rl.offline_rl.custom_envs.custom_envs_registration import (
    CustomEnv, RenderMode, register_grid_envs)
from training_rl.offline_rl.custom_envs.utils import \
    InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    MinariDatasetConfig
from training_rl.offline_rl.scripts.visualizations.utils import (
    get_state_action_data_and_policy_grid_distributions, snapshot_env)
from training_rl.offline_rl.utils import (load_buffer_minari,
                                          state_action_histogram)

NAME_EXPERT_DATA = "Grid_2D_8x8_discrete-combined_data_sets_A_B-V0"
ENV_NAME = CustomEnv.Grid_2D_8x8_discrete
ENV_RENDER_MODE = RenderMode.RGB_ARRAY_LIST  # Only for snapshot of environment


data_config = MinariDatasetConfig.load_from_file(NAME_EXPERT_DATA)
env_config = data_config.initial_config_2d_grid_env

register_grid_envs()
env = InitialConfigCustom2DGridEnvWrapper(
    gym.make(ENV_NAME, render_mode=ENV_RENDER_MODE), env_config=env_config
)


data = load_buffer_minari(NAME_EXPERT_DATA)

print(f"number of elements: {len(data)}")

state_action_count_data, _ = get_state_action_data_and_policy_grid_distributions(data, env)

state_action_histogram(state_action_count_data)

if ENV_RENDER_MODE:
    snapshot_env(env)
