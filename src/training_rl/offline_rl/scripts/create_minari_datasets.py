import minari

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import \
    BehaviorPolicyType
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    CustomEnv
from training_rl.offline_rl.custom_envs.utils import Grid2DInitialConfig
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import (
    MinariDatasetConfig, create_minari_config, create_minari_datasets)
from training_rl.offline_rl.generate_custom_minari_datasets.utils import (
    generate_compatible_minari_dataset_name, get_dataset_name_2d_grid)
from training_rl.offline_rl.load_env_variables import load_env_variables

load_env_variables()

ENV_NAME = CustomEnv.Grid_2D_8x8_continuous
# ENV_NAME = "Ant-v2" # or any other gym environment
DATA_SET_NAME = "data"
DATA_SET_IDENTIFIER = "move_around_obstacle"
VERSION_DATA_SET = "v0"
NUM_STEPS = 3000
BEHAVIOR_POLICY_NAME = BehaviorPolicyType.random

INITIAL_CONDITIONS_2D_GRID = {
    "obstacles": ObstacleTypes.vertical_object_8x8,
    "initial_state": (0, 0),
    "target_state": (0, 7),
}
initial_condition_2d_grid = Grid2DInitialConfig(**INITIAL_CONDITIONS_2D_GRID)

minari_dataset_config = create_minari_config(
    env_name=ENV_NAME,
    dataset_name=DATA_SET_NAME,
    data_set_identifier=DATA_SET_IDENTIFIER,
    version_dataset=VERSION_DATA_SET,
    num_steps=NUM_STEPS,
    behavior_policy_name=BEHAVIOR_POLICY_NAME,
    env_2d_grid_initial_config=initial_condition_2d_grid,
)

create_minari_datasets(minari_dataset_config)

data = minari.load_dataset(minari_dataset_config.data_set_name)
print("number of episodes collected: ", len(data))
# for elem in data:
#    print(elem.actions, elem.truncations, elem.terminations)
