import gymnasium as gym

import tianshou
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    register_grid_envs
from training_rl.offline_rl.custom_envs.utils import \
    InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.offline_policies.policy_registry import \
    PolicyFactoryRegistry
from training_rl.offline_rl.offline_trainings.policy_config_data_class import \
    TrainedPolicyConfig


def restore_trained_offline_policy(
    offline_policy_config: TrainedPolicyConfig,
) -> tianshou.policy.BasePolicy:
    register_grid_envs()
    env_name = offline_policy_config.minari_dataset_config.env_name
    render_mode = offline_policy_config.render_mode
    env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

    env = InitialConfigCustom2DGridEnvWrapper(
        gym.make(env_name, render_mode=render_mode), env_config=env_config
    )

    # Policy restoration
    policy_type = offline_policy_config.policy_name
    policy_config = offline_policy_config.policy_config
    policy = PolicyFactoryRegistry.__dict__[policy_type](
        policy_config=policy_config,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )

    return policy
