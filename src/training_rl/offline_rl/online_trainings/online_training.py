import os

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger

from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    register_grid_envs
from training_rl.offline_rl.custom_envs.utils import \
    InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.offline_policies.policy_registry import (
    PolicyFactoryRegistry, PolicyType)
from training_rl.offline_rl.offline_trainings.custom_tensorboard_callbacks import \
    CustomSummaryWriter
from training_rl.offline_rl.offline_trainings.offline_training import (
    create_policy, get_environments, setup_random_seed)
from training_rl.offline_rl.offline_trainings.policy_config_data_class import (
    TrainedPolicyConfig, get_trained_policy_path)

"""
def online_training(
    trained_policy_config: TrainedPolicyConfig,
    policy_type: PolicyType,
    num_epochs=1,
    batch_size=64,
    buffer_size=100000,
    step_per_epoch=100000,
    step_per_collect=10,
    repeat_per_collect=10,
    number_test_envs=5,
    number_train_envs=10,
    exploration_noise=True,
    episode_per_test=10,
    frames_stack=1,
    seed=None,
    restore_training=False,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environments
    register_grid_envs()
    env_name = trained_policy_config.minari_dataset_config.env_name
    render_mode = trained_policy_config.render_mode
    env_config = trained_policy_config.minari_dataset_config.initial_config_2d_grid_env

    env = InitialConfigCustom2DGridEnvWrapper(
        gym.make(env_name, render_mode=render_mode), env_config=env_config
    )

    test_envs = SubprocVectorEnv(
        [
            lambda: InitialConfigCustom2DGridEnvWrapper(gym.make(env_name), env_config=env_config)
            for _ in range(number_test_envs)
        ]
    )

    train_envs = SubprocVectorEnv(
        [
            lambda: InitialConfigCustom2DGridEnvWrapper(gym.make(env_name), env_config=env_config)
            for _ in range(number_train_envs)
        ]
    )

    # Policy restoration
    policy_name = trained_policy_config.policy_name
    policy_config = trained_policy_config.policy_config

    policy = PolicyFactoryRegistry.__dict__[policy_name](
        policy_config=policy_config,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )

    # Path to save models/config
    name_expert_data = trained_policy_config.name_expert_data
    log_name = os.path.join(name_expert_data, policy_name)
    log_path = get_trained_policy_path(log_name)

    if restore_training:
        policy_path = os.path.join(log_path, "policy.pth")
        policy.load_state_dict(torch.load(policy_path, map_location=trained_policy_config.device))
        print("Loaded policy from: ", policy_path)

    # Create collector for testing

    if number_train_envs > 1:
        # buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        buffer = VectorReplayBuffer(
            total_size=buffer_size,
            buffer_num=len(train_envs),
            stack_num=frames_stack,
            # ignore_obs_next=True,
            # save_only_last_obs=True,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=buffer_size,
            buffer_num=len(train_envs),
            stack_num=frames_stack,
            # ignore_obs_next=True,
            # save_only_last_obs=True,
        )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

    # Tensorboard writer
    custom_writer = CustomSummaryWriter(log_path, env)
    custom_writer.log_custom_info()
    logger = TensorboardLogger(custom_writer)

    trainer = None
    if policy_type == PolicyType.offpolicy:
        trainer = offpolicy_trainer
    elif policy_type == PolicyType.onpolicy:
        trainer = onpolicy_trainer

    # Training
    _ = trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=episode_per_test,
        batch_size=batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    # Save final policy
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

    # Save config
    trained_policy_config.save_to_file()

"""


def online_training(
    trained_policy_config: TrainedPolicyConfig,
    policy_type: PolicyType,
    num_epochs=1,
    batch_size=64,
    buffer_size=100000,
    step_per_epoch=100000,
    step_per_collect=10,
    repeat_per_collect=10,
    number_test_envs=5,
    number_train_envs=10,
    exploration_noise=True,
    episode_per_test=10,
    frames_stack=1,
    seed=None,
    restore_training=False,
):
    setup_random_seed(seed)
    # Create environments
    env, test_envs, train_envs = get_environments(
        policy_config=trained_policy_config,
        number_test_envs=number_test_envs,
        number_train_envs=number_train_envs,
    )

    # Path to save models/config
    policy_name = trained_policy_config.policy_name
    name_expert_data = trained_policy_config.name_expert_data
    log_name = os.path.join(name_expert_data, policy_name)
    log_path = get_trained_policy_path(log_name)

    # Policy creation/restoration
    policy = create_policy(policy_config=trained_policy_config, env=env)
    if restore_training:
        policy_path = os.path.join(log_path, "policy.pth")
        policy.load_state_dict(torch.load(policy_path, map_location=trained_policy_config.device))
        print("Loaded policy from: ", policy_path)

    # Create collector for testing

    if number_train_envs > 1:
        # buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        buffer = VectorReplayBuffer(
            total_size=buffer_size,
            buffer_num=len(train_envs),
            stack_num=frames_stack,
            # ignore_obs_next=True,
            # save_only_last_obs=True,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=buffer_size,
            buffer_num=len(train_envs),
            stack_num=frames_stack,
            # ignore_obs_next=True,
            # save_only_last_obs=True,
        )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

    # Tensorboard writer
    custom_writer = CustomSummaryWriter(log_path, env)
    custom_writer.log_custom_info()
    logger = TensorboardLogger(custom_writer)

    trainer = None
    if policy_type == PolicyType.offpolicy:
        trainer = offpolicy_trainer
    elif policy_type == PolicyType.onpolicy:
        trainer = onpolicy_trainer

    # Training
    _ = trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=episode_per_test,
        batch_size=batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    # Save final policy
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

    # Save config
    trained_policy_config.save_to_file()
