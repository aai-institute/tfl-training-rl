from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import tianshou.utils.net.discrete
import torch
from tianshou.policy import DiscreteBCQPolicy
from tianshou.utils.net.common import ActorCritic
from torch import nn

from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 0.0001,
    "gamma": 0.99,
    "n_step": 5,
    "target_update_freq": 50,
    "eps_test": 0.001,
    "unlikely_action_threshold": 0.4,  # the lower the most optimal policies but more o.o.d. data too!
    "imitation_logits_penalty": 0.001,  # 0.001
}


def bcq_discrete_default_config():
    return policy_config


def create_bcq_discrete_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    class DQNVector(nn.Module):
        def __init__(
            self,
            input_dim: int,
            action_shape: int,
            device: Union[str, int, torch.device] = "cpu",
            features_only: bool = False,
            output_dim: Optional[int] = None,
            layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        ) -> None:
            super().__init__()

            self.device = device
            self.net = nn.Sequential(
                layer_init(
                    nn.Linear(input_dim, 128)
                ),  # Adjust input_dim and hidden layers as needed
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, 128)),
                nn.ReLU(inplace=True),
            )
            with torch.no_grad():
                self.output_dim = np.prod(self.net(torch.zeros(1, input_dim)).shape[1:])
            if not features_only:
                self.net = nn.Sequential(
                    self.net,
                    layer_init(
                        nn.Linear(self.output_dim, 128)
                    ),  # Adjust output_dim and hidden layers as needed
                    nn.ReLU(inplace=True),
                    layer_init(nn.Linear(128, np.prod(action_shape))),
                )
                self.output_dim = np.prod(action_shape)
            elif output_dim is not None:
                self.net = nn.Sequential(
                    self.net,
                    layer_init(nn.Linear(self.output_dim, output_dim)),
                    nn.ReLU(inplace=True),
                )
                self.output_dim = output_dim

        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
        ) -> Tuple[torch.Tensor, Any]:
            r"""Mapping: s -> Q(s, \*)."""
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            return self.net(obs), state

    ## Define model to train
    device = "cpu"
    feature_net = DQNVector(
        observation_shape,
        action_shape,
        device=device,
        features_only=False,  # output_dim=2
    ).to(device)

    hidden_sizes = [512]
    policy_net = tianshou.utils.net.discrete.Actor(
        feature_net,
        action_shape,
        device=device,
        hidden_sizes=hidden_sizes,
        softmax_output=False,
        # preprocess_net_output_dim=action_shape,
    ).to(device)

    imitation_net = tianshou.utils.net.discrete.Actor(
        feature_net,
        action_shape,
        device=device,
        hidden_sizes=hidden_sizes,
        softmax_output=False,
        # preprocess_net_output_dim=action_shape,
    ).to(device)

    actor_critic = ActorCritic(policy_net, imitation_net)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=policy_config["lr"])

    policy = DiscreteBCQPolicy(
        model=policy_net,
        imitator=imitation_net,
        optim=optim,
        action_space=action_space,
        discount_factor=policy_config["gamma"],
        estimation_step=policy_config["n_step"],
        target_update_freq=policy_config["target_update_freq"],
        eval_eps=policy_config["eps_test"],
        unlikely_action_threshold=policy_config["unlikely_action_threshold"],
        imitation_logits_penalty=policy_config["imitation_logits_penalty"],
    )

    # policy.set_eps(policy_config["eps_test"])

    return policy
