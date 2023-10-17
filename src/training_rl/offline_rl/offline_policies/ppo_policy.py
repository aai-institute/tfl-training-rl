from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "lr": 3e-4,
    "gamma": 0.99,
    "device": "cpu",
    "hidden_sizes": [64, 64],
    "vf_coef": 0.25,
    "ent_coef": 0.0,
    "gae_lambda": 0.95,
    "bound_action_method": "clip",
    "max_grad_norm": 0.5,
    "eps_clip": 0.2,
    "dual_clip": None,
    "value_clip": 0,
    "norm_adv": 0,
    "recompute_adv": 0,
    "rew_norm": True,
    "lr_decay": {
        "step_per_epoch": None,
        "step_per_collect": None,
        "epoch": None,
    },
}


def ppo_default_config():
    return policy_config


def create_ppo_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    print(action_shape, observation_shape)

    # model
    net_a = Net(
        observation_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        activation=nn.Tanh,
        device=policy_config["device"],
    )
    actor = ActorProb(
        net_a,
        action_shape,
        unbounded=True,
        device=policy_config["device"],
    ).to(policy_config["device"])
    net_c = Net(
        observation_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        activation=nn.Tanh,
        device=policy_config["device"],
    )
    critic = Critic(net_c, device=policy_config["device"]).to(policy_config["device"])
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=policy_config["lr"])

    lr_scheduler = None
    if (
        policy_config["lr_decay"]["step_per_epoch"] is not None
        and policy_config["lr_decay"]["step_per_collect"] is not None
        and policy_config["lr_decay"]["epoch"] is not None
    ):
        # decay learning rate to 0 linearly
        max_update_num = (
            np.ceil(
                policy_config["lr_decay"]["step_per_epoch"]
                / policy_config["lr_decay"]["step_per_collect"]
            )
            * policy_config["lr_decay"]["epoch"]
        )

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=policy_config["gamma"],
        gae_lambda=policy_config["gae_lambda"],
        max_grad_norm=policy_config["max_grad_norm"],
        vf_coef=policy_config["vf_coef"],
        ent_coef=policy_config["ent_coef"],
        reward_normalization=policy_config["rew_norm"],
        action_scaling=True,
        action_bound_method=policy_config["bound_action_method"],
        lr_scheduler=lr_scheduler,
        action_space=action_space,
        eps_clip=policy_config["eps_clip"],
        value_clip=policy_config["value_clip"],
        dual_clip=policy_config["dual_clip"],
        advantage_normalization=policy_config["norm_adv"],
        recompute_advantage=policy_config["recompute_adv"],
    )

    return policy
