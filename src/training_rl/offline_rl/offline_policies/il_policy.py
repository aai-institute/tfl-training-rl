from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.policy import ImitationPolicy
from torch import nn

from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 0.001,
}


def il_default_config():
    return policy_config


class DQNVector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: int,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjust input_dim and hidden layers as needed
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(64, action_shape),
            nn.Softmax(),
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


def create_il_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    device = "cpu"

    net = DQNVector(observation_shape, action_shape, device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])
    policy = ImitationPolicy(actor=net, optim=optim, action_space=action_space)

    return policy
