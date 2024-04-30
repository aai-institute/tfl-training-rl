import minari
import os
from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType
from training_rl.offline_rl.custom_envs.custom_envs_registration import register_grid_envs
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    create_combined_minari_dataset
from training_rl.offline_rl.load_env_variables import load_env_variables

load_env_variables()
register_grid_envs()

if __name__ == '__main__':

    ENV_NAME = "torcs"

    DATA_SET_IDENTIFIER_I = "torcs_expert"
    NUM_STEPS_I = 100
    BEHAVIOR_POLICY_I = BehaviorPolicyType.torcs_expert_policy

    DATA_SET_IDENTIFIER_II = "torcs_suboptimal"
    NUM_STEPS_II = 6000
    BEHAVIOR_POLICY_II = BehaviorPolicyType.torcs_expert_policy_with_noise

    # ToDo:
    #  1 . Add Minari data collection
    #  2. Load in tianshou and do BC or CQL, etc.

    config_minari_data = create_combined_minari_dataset(
        env_name=ENV_NAME,
        dataset_names=(DATA_SET_IDENTIFIER_I, DATA_SET_IDENTIFIER_II),
        num_collected_points=(NUM_STEPS_I, NUM_STEPS_II),
        behavior_policy_names=(BEHAVIOR_POLICY_I, BEHAVIOR_POLICY_II),
        combined_dataset_identifier="torcs_driver_expert_and_suboptimal",
    )

    #data = minari.load_dataset("torcs_lidar_env-torcs_driver_expert_and_suboptimal-v0")
    os.system('pkill torcs')
