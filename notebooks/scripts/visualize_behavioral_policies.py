from training_rl.offline_rl.behavior_policies.behavior_policy_registry import \
    BehaviorPolicyType
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_envs_registration import (
    CustomEnv, RenderMode, register_grid_envs)
from training_rl.offline_rl.custom_envs.utils import Grid2DInitialConfig
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.offpolicy_rendering import \
    offpolicy_rendering

load_env_variables()

register_grid_envs()

ENV_NAME = CustomEnv.Grid_2D_8x8_discrete
# ENV_NAME = "Ant-v2" # Or any other gym environment.

BEHAVIOR_POLICY = BehaviorPolicyType.behavior_8x8_avoid_vertical_obstacle

# Only if we want to change the default initial conditions

env_2D_grid_initial_config = Grid2DInitialConfig(
    obstacles=ObstacleTypes.vertical_object_8x8,
    initial_state=(0, 0),
    target_state=(0, 7),
)

offpolicy_rendering(
    env_or_env_name=ENV_NAME,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    behavior_policy_name=BEHAVIOR_POLICY,
    env_2d_grid_initial_config=env_2D_grid_initial_config,
    num_frames=1000,
)
