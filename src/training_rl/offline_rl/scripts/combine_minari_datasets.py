# In this script we create a single datasets from two existing ones.

import minari
from minari import combine_datasets

from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    CustomEnv
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    MinariDatasetConfig
from training_rl.offline_rl.generate_custom_minari_datasets.utils import \
    generate_compatible_minari_dataset_name
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.utils import (check_minari_files_exist,
                                          delete_minari_data_if_exists)

load_env_variables()
minari_dataset_1 = (
    "Grid_2D_8x8_discrete-data_vertical_object_8x8_start_0_0_target_0_7move_around_obstacle-v0"
)
minari_dataset_2 = (
    "Grid_2D_8x8_discrete-data_vertical_object_8x8_start_0_0_target_0_7move_around_obstacle-v0"
)
NAME_COMBINED_DATASET = "combined_data_sets_vertical_obstacle"


list_dataset_names = [minari_dataset_1, minari_dataset_2]

check_minari_files_exist(list_dataset_names)

minari_datasets = [minari.load_dataset(dataset_id) for dataset_id in list_dataset_names]

name_combined_dataset = generate_compatible_minari_dataset_name(
    env_name=CustomEnv.Grid_2D_8x8_discrete, data_set_name=NAME_COMBINED_DATASET, version="v0"
)

delete_minari_data_if_exists(name_combined_dataset)
combined_dataset = combine_datasets(minari_datasets, new_dataset_id=name_combined_dataset)
print(
    f"Number of episodes in dataset A:{len(minari_datasets[0])}, in dataset B:{len(minari_datasets[1])} and  "
    f"in combined dataset: {len(combined_dataset)}"
)

minari_combined_dataset = MinariDatasetConfig.load_from_file(minari_dataset_1)
minari_combined_dataset.data_set_name = name_combined_dataset

# Save grid configuration
minari_combined_dataset.initial_config_2d_grid_env.initial_state_state = (0, 0)
minari_combined_dataset.initial_config_2d_grid_env.target_state = (0, 7)
minari_combined_dataset.save_to_file()
