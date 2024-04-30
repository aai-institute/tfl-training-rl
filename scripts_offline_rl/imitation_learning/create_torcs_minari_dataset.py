from training_rl.offline_rl.load_env_variables import load_env_variables

load_env_variables()

import os
from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    create_combined_minari_dataset

if __name__ == '__main__':
    from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
    env = EnvFactory.torcs.get_env()

    ENV_NAME = "torcs"

    DATA_SET_NAME_II = "torcs_suboptimal"
    NUM_STEPS_II = 4000
    BEHAVIOR_POLICY_II = BehaviorPolicyType.torcs_expert_policy_with_noise

    DATA_SET_NAME_I = "torcs_expert"
    NUM_STEPS_I = 2000
    BEHAVIOR_POLICY_I = BehaviorPolicyType.torcs_expert_policy

    config_minari_data = create_combined_minari_dataset(
        env_name=ENV_NAME,
        dataset_names=(DATA_SET_NAME_II, DATA_SET_NAME_I),
        num_collected_points=(NUM_STEPS_II, NUM_STEPS_I),
        behavior_policy_names=(BEHAVIOR_POLICY_II, BEHAVIOR_POLICY_I),
        combined_dataset_identifier="torcs_driver_expert_and_suboptimal",
    )

    #data = minari.load_dataset("torcs_lidar_env-torcs_driver_expert_and_suboptimal-v0")
    os.system('pkill torcs')
