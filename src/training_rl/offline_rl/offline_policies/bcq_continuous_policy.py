from typing import Any, Dict

import gymnasium as gym
import torch

from tianshou.policy import BCQPolicy
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import VAE, Critic, Perturbation
from training_rl.offline_rl.utils import extract_dimension

policy_config = {
    "hidden_sizes": [256, 256],
    "device": "cpu",
    "phi": 0.05,  # 0.05
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "vae_hidden_sizes": [512, 512],
    "gamma": 0.99,
    "tau": 0.005,
    "lmbda": 0.75,
    "forward_sampled_times": 100,
    "num_sampled_action": 10,
}


def bcq_continuous_default_config():
    return policy_config


def create_bcq_continuous_policy_from_dict(
    policy_config: Dict[str, Any],
    action_space: gym.core.ActType,
    observation_space: gym.core.ObsType,
):
    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)

    print(f"observation/action shapes {observation_shape, action_shape}")

    # model
    # perturbation network
    net_a = MLP(
        input_dim=observation_shape + action_shape,
        output_dim=action_shape,
        hidden_sizes=policy_config["hidden_sizes"],
        device=policy_config["device"],
    )

    max_action = action_space.high[0]
    actor = Perturbation(
        net_a, max_action=max_action, device=policy_config["device"], phi=policy_config["phi"]
    ).to(policy_config["device"])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=policy_config["actor_lr"])

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
    critic1 = Critic(net_c1, device=policy_config["device"]).to(policy_config["device"])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=policy_config["critic_lr"])
    critic2 = Critic(net_c2, device=policy_config["device"]).to(policy_config["device"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=policy_config["critic_lr"])

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=observation_shape + action_shape,
        hidden_sizes=policy_config["vae_hidden_sizes"],
        device=policy_config["device"],
    )

    latent_dim = action_shape * 2
    vae_decoder = MLP(
        input_dim=observation_shape + latent_dim,
        output_dim=action_shape,
        hidden_sizes=policy_config["vae_hidden_sizes"],
        device=policy_config["device"],
    )
    vae = VAE(
        vae_encoder,
        vae_decoder,
        hidden_dim=policy_config["vae_hidden_sizes"][-1],
        latent_dim=latent_dim,
        max_action=max_action,
        device=policy_config["device"],
    ).to(policy_config["device"])
    vae_optim = torch.optim.Adam(vae.parameters())

    policy = BCQPolicy(
        actor_perturbation=actor,
        actor_perturbation_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        vae=vae,
        vae_optim=vae_optim,
        device=policy_config["device"],
        gamma=policy_config["gamma"],
        tau=policy_config["tau"],
        lmbda=policy_config["lmbda"],
        forward_sampled_times=policy_config["forward_sampled_times"],
        num_sampled_action=policy_config["num_sampled_action"],
        action_space=action_space,
    )

    return policy
