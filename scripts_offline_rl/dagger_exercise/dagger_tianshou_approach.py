import os

import minari
import gymnasium as gym
import torch

from training_rl.offline_rl.custom_envs.custom_envs_registration import register_grid_envs
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.offpolicy_rendering import offpolicy_rendering
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.offline_training import offline_training
from training_rl.offline_rl.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from training_rl.offline_rl.offline_trainings.restore_policy_model import restore_trained_offline_policy
from training_rl.offline_rl.utils import load_buffer_minari

DATA_SET_NAME = "torcs_lidar_env-torcs_driver_expert_and_suboptimal-v0"
load_env_variables()
register_grid_envs()



if __name__ == '__main__':

    TRAIN = False

    #data = minari.load_dataset(DATA_SET_NAME)
    buffer_data = load_buffer_minari(DATA_SET_NAME)

    data_size = len(buffer_data)

    data_config = MinariDatasetConfig.load_from_file(DATA_SET_NAME)
    env_config = data_config.initial_config_2d_grid_env

    POLICY_NAME = PolicyName.imitation_learning_torcs

    NUM_EPOCHS = 20

    BATCH_SIZE = 128
    STEP_PER_EPOCH = 0.3 * data_size

    # After every epoch we will collect some test statistics from the policy from NUMBER_TEST_ENVS independent envs.
    NUMBER_TEST_ENVS = 1
    EXPLORATION_NOISE = True
    SEED = None  # 1626

    
    # TrainedPolicyConfig is a handy object that will help us to deal with the policy configuration data.
    offline_policy_config = TrainedPolicyConfig(
        name_expert_data=DATA_SET_NAME,
        policy_name=POLICY_NAME,
        #render_mode=render_mode,
        device="cpu"
    )

    if TRAIN:
        offline_training(
            offline_policy_config=offline_policy_config,
            num_epochs=NUM_EPOCHS,
            number_test_envs=NUMBER_TEST_ENVS,
            step_per_epoch=STEP_PER_EPOCH,
            restore_training=False,
            batch_size=BATCH_SIZE
        )
    else:
        NUM_STEPS = 10000
        POLICY_FILE = "policy_best_reward.pth"
        # restore a policy with the same configuration as the one we trained.
        policy = restore_trained_offline_policy(offline_policy_config)
        # load the weights
        name_expert_data = offline_policy_config.name_expert_data
        log_name = os.path.join(name_expert_data, POLICY_NAME)
        log_path = get_trained_policy_path(log_name)
        policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_FILE), map_location="cpu"))

        env = gym.make(data_config.env_name)


        offpolicy_rendering(
            env_or_env_name=env,
            render_mode=None,
            policy_model=policy,
            num_frames=NUM_STEPS,
        )

        env.end()
