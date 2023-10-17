from typing import Any, Dict

import gymnasium as gym
import tianshou
import torch
from tianshou.utils.net.common import Net

from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 0.0001,
    "gamma": 0.99,
    "device": "cpu",
    "hidden_sizes": [256, 256, 256],
    "n_steps": 3,
    "target_freq": 500,
    "epsilon": 1.0,  # exploration noise
}


def dqn_default_config():
    return policy_config


def create_dqn_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    print(action_shape, observation_shape)

    device = policy_config["device"]

    net = Net(
        state_shape=observation_shape,
        action_shape=action_shape,
        hidden_sizes=policy_config["hidden_sizes"],
    )
    optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])

    policy = tianshou.policy.DQNPolicy(
        net,
        optim,
        policy_config["gamma"],
        policy_config["n_steps"],
        target_update_freq=policy_config["target_freq"],
    )
    policy.set_eps(policy_config["epsilon"])

    return policy
