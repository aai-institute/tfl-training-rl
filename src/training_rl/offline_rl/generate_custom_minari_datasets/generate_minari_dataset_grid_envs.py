from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List, Sequence
import minari
import numpy as np
from minari import DataCollectorV0, combine_datasets
from minari.data_collector.callbacks import StepDataCallback
from minari.storage import get_dataset_path

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import (
    BehaviorPolicyRestorationConfigFactoryRegistry, BehaviorPolicyType)
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    register_grid_envs, EnvFactory
from training_rl.offline_rl.custom_envs.gym_torcs.gym_torcs import TorcsEnv, TorcsLidarEnv
from training_rl.offline_rl.custom_envs.utils import (
    Grid2DInitialConfig, InitialConfigCustom2DGridEnvWrapper)
from training_rl.offline_rl.generate_custom_minari_datasets.utils import (
    generate_compatible_minari_dataset_name, get_dataset_name_2d_grid)
from training_rl.offline_rl.utils import delete_minari_data_if_exists

OVERRIDE_DATA_SET = True


@dataclass
class MinariDatasetConfig:
    env_name: str
    data_set_name: str
    num_steps: int
    behavior_policy: BehaviorPolicyType = None
    initial_config_2d_grid_env: Grid2DInitialConfig = None
    children_dataset_names: List[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)

    def save_to_file(self):
        data_set_path = get_dataset_path(self.data_set_name)
        file_name = "config.json"

        obj_to_saved = asdict(self)
        if self.initial_config_2d_grid_env is not None:
            obj_to_saved["initial_config_2d_grid_env"]["obstacles"] = obj_to_saved[
                "initial_config_2d_grid_env"
            ]["obstacles"].value

        with open(os.path.join(data_set_path, file_name), "w") as file:
            json.dump(obj_to_saved, file, indent=4)

    @classmethod
    def load_from_file(cls, dataset_id):
        filename = get_dataset_path(dataset_id)
        with open(os.path.join(filename, "config.json"), "r") as file:
            config_dict = json.load(file)

        if config_dict["initial_config_2d_grid_env"] is not None:
            config_dict["initial_config_2d_grid_env"] = Grid2DInitialConfig(
                **config_dict["initial_config_2d_grid_env"]
            )

            config_dict["initial_config_2d_grid_env"].obstacles = ObstacleTypes(
                config_dict["initial_config_2d_grid_env"].obstacles
            )

        return cls(**config_dict)


def create_minari_collector_env_wrapper(
    env_name: str,
    initial_config_2d_grid_env: Grid2DInitialConfig = None,
):
    """
    Creates a wrapper 'DataCollectorV0' around the environment in order to collect data for minari dataset

    :param env_name:
    :param initial_config_2d_grid_env:
    :return:
    """

    env = EnvFactory[env_name].get_env(
        grid_config=initial_config_2d_grid_env
    )

    class CustomSubsetStepDataCallback(StepDataCallback):
        def __call__(self, env, **kwargs):
            step_data = super().__call__(env, **kwargs)
            # del step_data["observations"]["achieved_goal"]
            return step_data

    env = DataCollectorV0(
        env,
        step_data_callback=CustomSubsetStepDataCallback,
        record_infos=False,
    )
    return env


def create_minari_config(
    env_name: str,
    dataset_name: str,
    dataset_identifier: str,
    version_dataset: str,
    num_steps: int,
    behavior_policy_name: str = "",
    env_2d_grid_initial_config: Grid2DInitialConfig = None,
) -> MinariDatasetConfig:
    name_expert_data = generate_compatible_minari_dataset_name(
        env_name, dataset_name, version_dataset
    )

    dataset_name += dataset_identifier

    dataset_config = {
        "env_name": env_name,
        "data_set_name": name_expert_data,
        "num_steps": num_steps,
        "behavior_policy": behavior_policy_name,
    }

    if env_2d_grid_initial_config is not None:
        dataset_config["initial_config_2d_grid_env"] = env_2d_grid_initial_config
        dataset_name = get_dataset_name_2d_grid(env_2d_grid_initial_config) + dataset_identifier
        name_expert_data = generate_compatible_minari_dataset_name(
            env_name, dataset_name, version_dataset
        )

    dataset_config["data_set_name"] = name_expert_data

    return MinariDatasetConfig.from_dict(dataset_config)


def create_minari_datasets(
    env_name: str,
    dataset_name: str = "data",
    dataset_identifier: str = "",
    version_dataset: str = "v0",
    num_colected_points: int = 1000,
    behavior_policy_name: BehaviorPolicyType = BehaviorPolicyType.random,
    env_2d_grid_initial_config: Grid2DInitialConfig = None,
) -> MinariDatasetConfig:
    """
    Creates a custom Minari dataset and save a MinariDatasetConfig metadata to file (see /data/offline_data).

    :param env_name:
    :param dataset_name:
    :param dataset_identifier:
    :param version_dataset:
    :param num_colected_points:
    :param behavior_policy_name: One of our registered behavioral policies (see behavior_policy_registry.py).
    :param env_2d_grid_initial_config: If the environment is of type Custom2DGridEnv, its initial config
    :return:
    :rtype:
    """
    dataset_config = create_minari_config(
        env_name=env_name,
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        version_dataset=version_dataset,
        num_steps=num_colected_points,
        behavior_policy_name=behavior_policy_name,
        env_2d_grid_initial_config=env_2d_grid_initial_config,
    )

    delete_minari_data_if_exists(dataset_config.data_set_name, override_dataset=OVERRIDE_DATA_SET)
    env = create_minari_collector_env_wrapper(
        dataset_config.env_name,
        dataset_config.initial_config_2d_grid_env,
    )
    state, _ = env.reset()

    num_steps = 0
    for _ in range(dataset_config.num_steps):
        behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[
            dataset_config.behavior_policy
        ]
        if dataset_config.behavior_policy == BehaviorPolicyType.random:
            action = env.action_space.sample()
        elif isinstance(env.unwrapped, TorcsEnv):
            raw_observation = env.raw_observation
            action = np.array(behavior_policy(raw_observation, env), dtype=np.float32)
        else:
            action = behavior_policy(state, env)

        next_state, reward, done, time_out, info = env.step(action)

        num_steps += 1

        if done or time_out:
            state, _ = env.reset()
            num_steps = 0
        else:
            state = next_state

    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_config.data_set_name, collector_env=env
    )
    dataset_config.save_to_file()

    if isinstance(env.unwrapped, TorcsLidarEnv):
        env.end()

    return dataset_config


# ToDo: Add a flag to keep or not the single datasets.
def create_combined_minari_dataset(
    env_name: str,
    dataset_names: Tuple[str, str] = ("data_I", "data_II"),
    dataset_identifiers: Tuple[str, str] = ("", ""),
    num_collected_points: Tuple[int, int] = (1000, 1000),
    behavior_policy_names: Tuple[BehaviorPolicyType, BehaviorPolicyType] = (
        BehaviorPolicyType.random,
        BehaviorPolicyType.random,
    ),
    combined_dataset_identifier: str = "combined_dataset",
    version_dataset: str = "v0",
    env_2d_grid_initial_config: Grid2DInitialConfig | Sequence[Grid2DInitialConfig] = None,
) -> MinariDatasetConfig:
    """
    Combine two minari datsets into a single one and save metadata with useful information.
    """
    collected_dataset_names = []

    if isinstance(env_2d_grid_initial_config, Grid2DInitialConfig) or env_2d_grid_initial_config is None:
        env_2d_grid_initial_config = [env_2d_grid_initial_config for _ in range(len(dataset_names))]
    else:
        if len(env_2d_grid_initial_config) != len(dataset_names):
            raise ValueError("env_2d_grid_initial_config must be of same length as dataset_names")

    for dataset_name, dataset_identifier, num_points, behavior_policy, env_2d_grid_initial_config_single in zip(
        dataset_names, dataset_identifiers, num_collected_points, behavior_policy_names, env_2d_grid_initial_config
    ):
        dataset_config = create_minari_datasets(
            env_name=env_name,
            dataset_name=dataset_name,
            dataset_identifier=dataset_identifier,
            num_colected_points=num_points,
            behavior_policy_name=behavior_policy,
            env_2d_grid_initial_config=env_2d_grid_initial_config_single,
        )

        collected_dataset_names.append(dataset_config.data_set_name)

    name_combined_dataset = generate_compatible_minari_dataset_name(
        env_name=env_name, data_set_name=combined_dataset_identifier, version=version_dataset
    )

    delete_minari_data_if_exists(name_combined_dataset)

    minari_datasets = [minari.load_dataset(dataset_id) for dataset_id in collected_dataset_names]

    combined_dataset = combine_datasets(minari_datasets, new_dataset_id=name_combined_dataset)

    print(
        f"Number of episodes in dataset I:{len(minari_datasets[0])}, in dataset II:{len(minari_datasets[1])} and  "
        f"in the combined dataset: {len(combined_dataset)}"
    )

    total_num_steps = int(np.sum(num_collected_points))

    # Create metadata for the combined dataset (we can reuse the metadata of set 0 for simplicity)
    minari_combined_dataset_config = MinariDatasetConfig.load_from_file(collected_dataset_names[0])
    minari_combined_dataset_config.num_steps = total_num_steps
    minari_combined_dataset_config.data_set_name = name_combined_dataset
    minari_combined_dataset_config.save_to_file()
    minari_combined_dataset_config.children_dataset_names = collected_dataset_names

    return minari_combined_dataset_config


# ToDo:
# 1 - Add info to Minari no possible yet: bug in Minari - issue open: https://github.com/Farama-Foundation/Minari/issues/125
