import os
import gymnasium as gym
import torch
from training_rl.offline_rl.custom_envs.custom_envs_registration import RenderMode
from training_rl.offline_rl.custom_envs.utils import InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.offpolicy_rendering import \
    offpolicy_rendering
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.policy_config_data_class import (
    TrainedPolicyConfig, get_trained_policy_path)
from training_rl.offline_rl.offline_trainings.restore_policy_model import \
    restore_trained_offline_policy

load_env_variables()

NAME_EXPERT_DATA = (
    "Grid_2D_8x8_discrete-data_verical_object_8x8_start_0_0_target_0_7move_around_obstacle-v0"
)
# "pen-cloned-v1"#"relocate-cloned-v1"
POLICY_TYPE = PolicyName.dqn
EXPLORATION_NOISE = False
EXPLORATION_EPSILON = 0.01
POLICY_NAME = "policy_final.pth"

offline_policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_TYPE,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu",
)

policy = restore_trained_offline_policy(offline_policy_config)

if EXPLORATION_NOISE:
    policy.set_eps(EXPLORATION_EPSILON)


name_expert_data = offline_policy_config.name_expert_data
log_name = os.path.join(name_expert_data, POLICY_TYPE)
log_path = get_trained_policy_path(log_name)
policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_NAME), map_location="cpu"))


env_name = offline_policy_config.minari_dataset_config.env_name
render_mode = offline_policy_config.render_mode

env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

# Change initial config
# env_config.obstacles = ObstacleTypes.obst_free_8x8
# env_config.initial_state = (0, 0)
# env_config.target_state = (0, 7)


env = InitialConfigCustom2DGridEnvWrapper(
    gym.make(env_name, render_mode=render_mode), env_config=env_config
)


offpolicy_rendering(
    env_or_env_name=env_name,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    policy_model=policy,
    env_2d_grid_initial_config=env_config,
    num_frames=1000,
)

# ToDo: Open issue in Minari as rendering is not working properly
# final_collector = Collector(policy, env, exploration_noise=EXPLORATION_NOISE)
# final_collector.collect(n_episode=20, render=1 / 35)
