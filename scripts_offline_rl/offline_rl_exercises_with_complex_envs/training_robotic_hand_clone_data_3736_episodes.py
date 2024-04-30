from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import gymnasium as gym
import torch
import os
from tianshou.policy import BasePolicy
from typing import Literal
from training_rl.offline_rl.custom_envs.custom_envs_registration import RenderMode
from training_rl.offline_rl.offline_policies.offpolicy_rendering import offpolicy_rendering
from training_rl.offline_rl.offline_trainings.restore_policy_model import restore_trained_offline_policy
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.offline_training import offline_training
from training_rl.offline_rl.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from training_rl.offline_rl.utils import load_buffer_minari


def restore_policy(
        offline_policy_config: TrainedPolicyConfig,
        policy_name: Literal["policy.pth", "policy_best_reward.pth"] = "policy_best_reward.pth",
) -> BasePolicy:
    # restore a policy with the same configuration as the one we trained.
    policy = restore_trained_offline_policy(offline_policy_config)
    # load the weights
    name_expert_data = offline_policy_config.name_expert_data
    log_name = os.path.join(name_expert_data, offline_policy_config.policy_name)
    log_path = get_trained_policy_path(log_name)
    policy.load_state_dict(torch.load(os.path.join(log_path, policy_name), map_location="cpu"))
    return policy


TRAIN = False
RESTORE_TRAINING = False

DATASET = 'pen-human-v2'
ENV_NAME = 'AdroitHandPen-v1'
NUM_EPOCHS = 4
NUMBER_TEST_ENVS = 1
STEP_PER_EPOCH = len(load_buffer_minari(DATASET))
POLICY_NAME = PolicyName.bcq_continuous


offline_policy_config = TrainedPolicyConfig(
    name_expert_data=DATASET,
    policy_name=POLICY_NAME,
    device="cuda"
)


if TRAIN:
    offline_training(
        offline_policy_config=offline_policy_config,
        num_epochs=NUM_EPOCHS,
        number_test_envs=NUMBER_TEST_ENVS,
        step_per_epoch=STEP_PER_EPOCH,
        restore_training=RESTORE_TRAINING,
    )
else:
    policy = restore_policy(offline_policy_config=offline_policy_config, policy_name="policy.pth")
    env = gym.make(ENV_NAME, max_episode_steps=1000, render_mode=RenderMode.HUMAN)

    offpolicy_rendering(
        env_or_env_name=env,
        render_mode=RenderMode.HUMAN,
        policy_model=policy,
        num_frames=100000,
    )
