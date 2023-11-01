import os

import gymnasium as gym
import torch

from training_rl.offline_rl.custom_envs.utils import \
    InitialConfigCustom2DGridEnvWrapper
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.policy_config_data_class import (
    TrainedPolicyConfig, get_trained_policy_path)
from training_rl.offline_rl.offline_trainings.restore_policy_model import \
    restore_trained_offline_policy
from training_rl.offline_rl.scripts.visualizations.utils import \
    get_state_action_data_and_policy_grid_distributions
from training_rl.offline_rl.utils import (compare_state_action_histograms,
                                          load_buffer_minari,
                                          state_action_histogram)

NAME_EXPERT_DATA = "Grid_2D_8x8_discrete-data_obst_free_8x8_start_0_0_target_7_7-v0"
# "Ant-v2-data-v0"
POLICY_NAME = PolicyName.imitation_learning
NUM_EPISODES = 200  # as more the more precise the statistics
POLICY_FILE = "policy.pth"
EXPLORATION_NOISE = True

offline_policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_NAME,
    device="cpu",
)

policy = restore_trained_offline_policy(offline_policy_config)
name_expert_data = offline_policy_config.name_expert_data
log_name = os.path.join(name_expert_data, POLICY_NAME)
log_path = get_trained_policy_path(log_name)
policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_FILE), map_location="cpu"))


env_name = offline_policy_config.minari_dataset_config.env_name
render_mode = offline_policy_config.render_mode

# 2d grid env configuration load from json
env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env
# env_config = Grid2DInitialConfig(
#    obstacles=ObstacleTypes.obst_free_8x8,
#    initial_state=(0, 0),
#    target_state=(7, 0),
# )


env = InitialConfigCustom2DGridEnvWrapper(
    gym.make(env_name, render_mode=render_mode), env_config=env_config
)


data = load_buffer_minari(NAME_EXPERT_DATA)

print(f"number of elements: {len(data)}")

(
    state_action_count_data,
    state_action_count_policy,
) = get_state_action_data_and_policy_grid_distributions(
    data, env, policy, num_episodes=NUM_EPISODES, logits_sampling=False
)
state_action_histogram(state_action_count_data)

state_action_histogram(state_action_count_policy)
compare_state_action_histograms(state_action_count_data, state_action_count_policy)
