import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    RenderMode
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    MinariDatasetConfig
from training_rl.offline_rl.offline_policies.policy_registry import (
    DefaultPolicyConfigFactoryRegistry, PolicyName)


def get_trained_policy_path(dataset_id):
    datasets_path = os.environ.get("TRAINED_POLICY_PATH")
    if datasets_path is not None:
        file_path = os.path.join(datasets_path, dataset_id)

    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)


@dataclass
class TrainedPolicyConfig:
    policy_name: PolicyName
    render_mode: RenderMode = None
    name_expert_data: str = None
    minari_dataset_config: MinariDatasetConfig = None
    policy_config: DefaultPolicyConfigFactoryRegistry = None
    device: str = "cpu"

    def __post_init__(self):
        if self.policy_config is None:
            self.policy_config = DefaultPolicyConfigFactoryRegistry.__dict__[self.policy_name]()
        self.minari_dataset_config = MinariDatasetConfig.load_from_file(self.name_expert_data)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)
        # return cls(
        #    policy_name=config_dict["policy_name"],
        #    render_mode=config_dict["render_mode"],
        #    policy_config=config_dict["policy_config"],
        #    minari_dataset_config=config_dict["minari_dataset_config"],
        # )

    def save_to_file(self):
        data_set_path = get_trained_policy_path(self.minari_dataset_config.data_set_name)
        file_name = "config.json"
        obj_to_save = asdict(self)

        if self.minari_dataset_config.initial_config_2d_grid_env is not None:
            obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ] = obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ].value
        with open(os.path.join(data_set_path, self.policy_name, file_name), "w") as file:
            json.dump(obj_to_save, file, indent=4)

    @classmethod
    def load_from_file(cls, dataset_id):
        filename = get_trained_policy_path(dataset_id)
        with open(os.path.join(filename, "config.json"), "r") as file:
            config_dict = json.load(file)

        config_dict["minari_dataset_config"] = MinariDatasetConfig.from_dict(
            config_dict["minari_dataset_config"]
        )
        return cls(**config_dict)


"""
config = {
    #"env_name": "ivan_bla",
    "name_expert_data": "Ant-v2-data-v0", #"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_2_2_target_7_7-v0",
    "policy_name": PolicyType.cql_continuous,
    "render_mode": RenderMode.RGB_ARRAY_LIST,
    "policy_config": DefaultPolicyConfigFactoryRegistry.bcq_discrete(),
    #"TAG": ""
}

#bla = OfflineTrainedPolicyConfig(**config)
##bla.save_to_file()
##bla.load_from_file(config["name_expert_data"])
#print(bla.policy_config())

"""
