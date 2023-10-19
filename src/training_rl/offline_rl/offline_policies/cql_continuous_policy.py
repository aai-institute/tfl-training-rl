from typing import Any, Dict

import gymnasium as gym
import numpy as np
import tianshou
import torch
from tianshou.policy import CQLPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 0.001,  # 6.25e-5,
    "gamma": 0.99,
    "hidden_sizes": [256],
    "critic_lr": 3e-4,
    "auto_alpha": True,
    "alpha_lr": 1e-4,
    "actor_lr": 1e-4,
    "alpha": 0.3,
    "cql_alpha_lr": 3e-4,
    "cql_weight": 1.0,  # change to 1.0!!!
    "tau": 0.005,  # 0.005
    "temperature": 1.0,
    "with_lagrange": True,
    "lagrange_threshold": 10.0,
    "device": "cpu",
}


def cql_continuous_default_config():
    return policy_config


def create_cql_continuous_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    # model
    # actor network
    net_a = Net(
        observation_shape,
        action_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        device=policy_config["device"],
    )
    actor = ActorProb(
        net_a,
        action_shape=[action_shape],
        device=policy_config["device"],
        unbounded=True,
        conditioned_sigma=True,
    ).to(policy_config["device"])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=policy_config["actor_lr"])

    # critic network
    net_c1 = Net(
        observation_shape,
        action_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        concat=True,
        device=policy_config["device"],
    )
    net_c2 = Net(
        observation_shape,
        action_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        concat=True,
        device=policy_config["device"],
    )
    critic1 = tianshou.utils.net.continuous.Critic(net_c1, device=policy_config["device"]).to(
        policy_config["device"]
    )
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=policy_config["critic_lr"])
    critic2 = tianshou.utils.net.continuous.Critic(net_c2, device=policy_config["device"]).to(
        policy_config["device"]
    )
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=policy_config["critic_lr"])

    if policy_config["auto_alpha"]:
        target_entropy = -np.prod(action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=policy_config["device"])
        alpha_optim = torch.optim.Adam([log_alpha], lr=policy_config["alpha_lr"])
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = policy_config["alpha"]

    policy = CQLPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        cql_alpha_lr=policy_config["cql_alpha_lr"],
        cql_weight=policy_config["cql_weight"],
        tau=policy_config["tau"],
        gamma=policy_config["gamma"],
        alpha=alpha,
        temperature=policy_config["temperature"],
        with_lagrange=policy_config["with_lagrange"],
        lagrange_threshold=policy_config["lagrange_threshold"],
        min_action=np.min(action_space.low),
        max_action=np.max(action_space.high),
        device=policy_config["device"],
    )

    return policy
