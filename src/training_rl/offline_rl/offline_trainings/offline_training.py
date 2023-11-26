import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import BasePolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger

from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    register_grid_envs
from training_rl.offline_rl.custom_envs.utils import \
    InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.offline_policies.policy_registry import \
    PolicyFactoryRegistry
from training_rl.offline_rl.offline_trainings.custom_tensorboard_callbacks import \
    CustomSummaryWriter
from training_rl.offline_rl.offline_trainings.policy_config_data_class import (
    TrainedPolicyConfig, get_trained_policy_path)
from training_rl.offline_rl.utils import load_buffer_minari

POLICY_NAME_BEST_REWARD = "policy_best_reward.pth"
POLICY_NAME = "policy.pth"


def setup_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_environments(
    policy_config: TrainedPolicyConfig,
    number_test_envs: Optional[int] = None,
    number_train_envs: Optional[int] = None,
) -> (gym.Env, SubprocVectorEnv):
    register_grid_envs()
    env_name = policy_config.minari_dataset_config.env_name
    render_mode = policy_config.render_mode
    env_config = policy_config.minari_dataset_config.initial_config_2d_grid_env

    env = InitialConfigCustom2DGridEnvWrapper(
        gym.make(env_name, render_mode=render_mode), env_config=env_config
    )

    test_envs = train_envs = None

    if number_test_envs is not None:
        test_envs = SubprocVectorEnv(
            [
                lambda: InitialConfigCustom2DGridEnvWrapper(
                    gym.make(env_name), env_config=env_config
                )
                for _ in range(number_test_envs)
            ]
        )

    if number_train_envs is not None:
        train_envs = SubprocVectorEnv(
            [
                lambda: InitialConfigCustom2DGridEnvWrapper(
                    gym.make(env_name), env_config=env_config
                )
                for _ in range(number_train_envs)
            ]
        )

    return env, test_envs, train_envs


def create_policy(policy_config: TrainedPolicyConfig, env: gym.Env) -> BasePolicy:
    policy_name = policy_config.policy_name
    policy_config = policy_config.policy_config
    policy = PolicyFactoryRegistry.__dict__[policy_name](
        policy_config=policy_config,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )

    return policy


def offline_training(
    offline_policy_config: TrainedPolicyConfig,
    step_per_epoch,
    step_per_collect=1,
    num_epochs=1,
    batch_size=64,
    update_per_step=1,
    number_test_envs=1,
    exploration_noise=True,
    restore_training=False,
    seed=None,
):
    """
    offline policy training with a Minari dataset. The policy could be one of the ones you can find in
    /offline_policies/policy_registry.py .

    :param step_per_epoch: the number of transitions collected per epoch.
    :param step_per_collect: the number of transitions the collector would collect before the network
        update, i.e., trainer will collect "step_per_collect" transitions and do some policy network update
        repeatedly in each epoch.
    :param offline_policy_config: an object of type TrainedPolicyConfig with all the necessary info for the training.
    :param num_epochs:
    :param batch_size: the batch size of sample data, which is going to feed in
        the policy network
    :param update_per_step: the number of policy network updates, so-called
        gradient steps, per epoch
    :param number_test_envs: the number of test_envs used to test the performance of the policy during training
    :param exploration_noise:
    :param restore_training:
    :param seed:
    :return:
    """
    setup_random_seed(seed)

    env, test_envs, _ = get_environments(offline_policy_config, number_test_envs)
    name_expert_data = offline_policy_config.name_expert_data
    data_buffer = load_buffer_minari(name_expert_data)

    # Path to save models/config
    policy_name = offline_policy_config.policy_name
    log_name = os.path.join(name_expert_data, policy_name)
    log_path = get_trained_policy_path(log_name)

    # Policy creation/restoration
    policy = create_policy(offline_policy_config, env)

    if restore_training:
        policy_path = os.path.join(log_path, POLICY_NAME)
        policy.load_state_dict(torch.load(policy_path, map_location=offline_policy_config.device))
        print("Loaded policy from: ", policy_path)

    # Create collector for testing
    test_collector = Collector(policy, test_envs, exploration_noise=exploration_noise)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, POLICY_NAME_BEST_REWARD))

    def stop_fn(mean_rewards):
        return False

    # Tensorboard writer
    custom_writer = CustomSummaryWriter(log_path, env)
    custom_writer.log_custom_info()
    logger = TensorboardLogger(custom_writer)

    # Training
    _ = OfflineTrainer(
        policy=policy,
        buffer=data_buffer,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        episode_per_test=number_test_envs,
        batch_size=batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    # Save final policy
    torch.save(policy.state_dict(), os.path.join(log_path, POLICY_NAME))

    # Save config
    offline_policy_config.save_to_file()
