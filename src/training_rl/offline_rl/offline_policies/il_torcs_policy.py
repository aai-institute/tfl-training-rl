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


def il_torcs_default_config():
    return policy_config


class ImitationLearningTorcs(nn.Module):
    def __init__(self, input_dim, action_shape=1, device: Union[str, int, torch.device] = "cpu",):
        super(ImitationLearningTorcs, self).__init__()

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),  # Adjust input_dim and hidden layers as needed
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, action_shape),
            nn.Tanh(),
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


def create_il_torcs_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    device = "cpu"

    net = ImitationLearningTorcs(observation_shape, action_shape).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])
    policy = ImitationPolicy(actor=net, optim=optim, action_space=action_space)

    return policy
