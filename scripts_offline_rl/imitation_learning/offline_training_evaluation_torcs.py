from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()


import os
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
    BehaviorPolicyRestorationConfigFactoryRegistry
from training_rl.offline_rl.custom_envs.custom_envs_registration import register_grid_envs, RenderMode, EnvFactory
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from training_rl.offline_rl.offline_policies.offpolicy_rendering import offpolicy_rendering
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.offline_training import offline_training
from training_rl.offline_rl.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from training_rl.offline_rl.offline_trainings.restore_policy_model import restore_trained_offline_policy
from training_rl.offline_rl.utils import load_buffer_minari

def compare_expert_vs_model_actions(array1, array2, label1="", label2="", title=""):
    if array1.shape != array2.shape:
        raise ValueError(f"Arrays have different shapes")

    x_values = np.arange(array1.size)
    plt.scatter(x_values, array1, label=label1)
    plt.scatter(x_values, array2, label=label2)

    # Highlight differing elements
    #differing_indices = np.where(array1 != array2)[0]
    #plt.scatter(differing_indices, array1[differing_indices], color='red', label='Different Element')
    #plt.scatter(differing_indices, array2[differing_indices], color='red')
    #plt.scatter(x_values, array1, marker="x")

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()

    # Plot histograms
    plt.hist(array1, bins=20, range=(-1, 1), alpha=0.5, label=label1)
    plt.hist(array2, bins=20, range=(-1, 1), alpha=0.5, label=label2)
    plt.xlabel('action (steer angle)')
    plt.ylabel('Frequency')
    plt.title('Comparison of actions of behavioral vs learned policies')
    plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


if __name__ == '__main__':

    TRAIN = False
    NUM_EPOCHS = 60
    BATCH_SIZE = 128
    NUMBER_TEST_ENVS = 1
    EXPLORATION_NOISE = True
    SEED = None  # 1626
    PERCENTAGE_DATA_PER_EPOCH = 1.0

    POLICY_NAME = PolicyName.imitation_learning_torcs
    #POLICY_NAME = PolicyName.cql_continuous

    #DATA_SET_NAME = "torcs-torcs_driver_expert_and_suboptimal-v0"
    DATA_SET_NAME = "torcs-torcs_expert_and_noise-v0"

    POLICY_NAME_TO_SAVED = "policy_bc_epoch_1.pt"

    buffer_data = load_buffer_minari(DATA_SET_NAME)
    data_config = MinariDatasetConfig.load_from_file(DATA_SET_NAME)


    offline_policy_config = TrainedPolicyConfig(
        name_expert_data=DATA_SET_NAME,
        policy_name=POLICY_NAME,
        device="cpu"
    )

    if TRAIN:
        offline_training(
            offline_policy_config=offline_policy_config,
            num_epochs=NUM_EPOCHS,
            number_test_envs=NUMBER_TEST_ENVS,
            step_per_epoch=PERCENTAGE_DATA_PER_EPOCH * len(buffer_data),
            restore_training=False,
            batch_size=BATCH_SIZE,
            policy_name=POLICY_NAME_TO_SAVED
        )

    else:
        POLICY_FILE = POLICY_NAME_TO_SAVED
        # restore a policy with the same configuration as the one we trained.
        policy = restore_trained_offline_policy(offline_policy_config)
        # load the weights
        name_expert_data = offline_policy_config.name_expert_data
        log_name = os.path.join(name_expert_data, POLICY_NAME)
        log_path = get_trained_policy_path(log_name)
        policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_FILE), map_location="cpu"))


        collected_observations = buffer_data.obs
        expert_collected_actions = buffer_data.act
        # Compare with BC policy action predictions
        policy_output = policy(Batch({"obs": buffer_data.obs, "info": {}})).act
        policy_collected_actions = policy_output if isinstance(policy_output, np.ndarray) else policy_output.detach().numpy()
        #policy_collected_actions = policy(Batch({"obs": buffer_data.obs, "info": {}})).act.detach().numpy()
        compare_expert_vs_model_actions(
            expert_collected_actions,
            policy_collected_actions,
            label1="behavioral_policy",
            label2="learned policy",
            title="Comparison",
        )

        offpolicy_rendering(
            env_or_env_name=data_config.env_name,
            render_mode=None,
            policy_model=policy,
            num_frames=10000,
        )
