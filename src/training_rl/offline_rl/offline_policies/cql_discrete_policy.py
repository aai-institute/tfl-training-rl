from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.policy import DiscreteCQLPolicy

from training_rl.offline_rl.offline_policies.il_policy import DQNVector
from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 0.0001,
    "gamma": 0.99,
    "n_step": 5,
    "target_update_freq": 50,
    "num_quantiles": 20,
    "min_q_weight": 13.0,
    "device": "cpu",
}


def cql_discrete_default_config():
    return policy_config


def create_cql_discrete_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    class QRDQN_simple(DQNVector):
        def __init__(
            self,
            observation_shape: int,
            action_shape: int,
            num_quantiles: int = 200,
            device: Union[str, int, torch.device] = "cpu",
        ) -> None:
            self.action_num = np.prod(action_shape)
            super().__init__(observation_shape, self.action_num * num_quantiles, device)
            self.num_quantiles = num_quantiles

        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
        ) -> Tuple[torch.Tensor, Any]:
            r"""Mapping: x -> Z(x, \*)."""
            obs, state = super().forward(obs)
            obs = obs.view(-1, self.action_num, self.num_quantiles)
            return obs, state

    net = QRDQN_simple(observation_shape, action_shape, policy_config["num_quantiles"])

    optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])
    policy = DiscreteCQLPolicy(
        model=net,
        optim=optim,
        action_space=action_space,
        discount_factor=policy_config["gamma"],
        num_quantiles=policy_config["num_quantiles"],
        estimation_step=policy_config["n_step"],
        target_update_freq=policy_config["target_update_freq"],
        min_q_weight=policy_config["min_q_weight"],
    ).to(policy_config["device"])

    return policy
