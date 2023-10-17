import json
import os
from dataclasses import asdict, dataclass
from typing import Dict

import gymnasium as gym
import minari
from minari import DataCollectorV0
from minari.data_collector.callbacks import StepDataCallback
from minari.storage import get_dataset_path

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import (
    BehaviorPolicyRestorationConfigFactoryRegistry, BehaviorPolicyType)
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    register_grid_envs
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
    env_name: str, initial_config_2d_grid_env: Grid2DInitialConfig = None
):
    """
    Creates a wrapper 'DataCollectorV0' around the environment in order to collect data for minari dataset

    :param env_name:
    :param initial_config_2d_grid_env:
    :return:
    """
    register_grid_envs()
    env = gym.make(env_name)
    env = InitialConfigCustom2DGridEnvWrapper(env, env_config=initial_config_2d_grid_env)

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


def create_minari_datasets(dataset_config: MinariDatasetConfig):
    """
    Creates a custom Minari dataset and save the MinariDatasetConfig metadata to file (see /data/offline_data).

    :param dataset_config:
    :return:
    """

    delete_minari_data_if_exists(dataset_config.data_set_name, override_dataset=OVERRIDE_DATA_SET)
    env = create_minari_collector_env_wrapper(
        dataset_config.env_name, dataset_config.initial_config_2d_grid_env
    )
    state, _ = env.reset()

    num_steps = 0
    for _ in range(dataset_config.num_steps):
        behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[
            dataset_config.behavior_policy
        ]
        if dataset_config.behavior_policy == BehaviorPolicyType.random:
            action = env.action_space.sample()
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


def create_minari_config(
    env_name: str,
    dataset_name: str,
    data_set_identifier: str,
    version_dataset: str,
    num_steps: int,
    behavior_policy_name: str = "",
    env_2d_grid_initial_config: Grid2DInitialConfig = None,
) -> MinariDatasetConfig:
    name_expert_data = generate_compatible_minari_dataset_name(
        env_name, dataset_name, version_dataset
    )

    dataset_name += data_set_identifier

    dataset_config = {
        "env_name": env_name,
        "data_set_name": name_expert_data,
        "num_steps": num_steps,
        "behavior_policy": behavior_policy_name,
    }

    if env_2d_grid_initial_config is not None:
        dataset_config["initial_config_2d_grid_env"] = env_2d_grid_initial_config
        dataset_name = get_dataset_name_2d_grid(env_2d_grid_initial_config) + data_set_identifier
        name_expert_data = generate_compatible_minari_dataset_name(
            env_name, dataset_name, version_dataset
        )

    dataset_config["data_set_name"] = name_expert_data

    return MinariDatasetConfig.from_dict(dataset_config)


# ToDo:
# 1 - Add info to Minari no possible yet: bug in Minari - issue open: https://github.com/Farama-Foundation/Minari/issues/125
