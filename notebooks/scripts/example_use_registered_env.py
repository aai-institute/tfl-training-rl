import gymnasium as gym

from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_envs_registration import (
    CustomEnv, RenderMode, register_grid_envs)
from training_rl.offline_rl.custom_envs.utils import (
    Grid2DInitialConfig, InitialConfigCustom2DGridEnvWrapper)

register_grid_envs()

ENV_NAME = CustomEnv.Grid_2D_8x8_continuous
# ENV_NAME = "Ant-v2" # Or any other gym environment.

RENDER_MODE = RenderMode.RGB_ARRAY_LIST

# Only if we want to change the default initial conditions
env_2d_grid_initial_config = Grid2DInitialConfig(
    obstacles=ObstacleTypes.obst_middle_8x8,
    initial_state=(0, 0),
    target_state=(7, 7),
)

env = InitialConfigCustom2DGridEnvWrapper(
    gym.make(ENV_NAME, render_mode=RENDER_MODE), env_config=env_2d_grid_initial_config
)
