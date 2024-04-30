# ToDo: load_env_variables should be done somewhere else
from typing import Dict
import gym
from training_rl.offline_rl.offline_policies.minimal_GPT_model import DecisionTransformer


decision_transformer_config = {
    "device": "cpu",
    "context_len": 20,
    "n_blocks": 3,
    "embed_dim": 128,
    "n_heads": 1,
    "dropout_p": 0.1,
}


def get_decision_transformer_default_config() -> Dict:
    return decision_transformer_config


def create_decision_transformer_policy_from_dict(
        config: Dict,
        action_space: gym.core.ActType,
        observation_space: gym.core.ObsType,
) -> DecisionTransformer:

    state_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=config["n_blocks"],
        h_dim=config["embed_dim"],
        context_len=config["context_len"],
        n_heads=config["n_heads"],
        drop_p=config["dropout_p"],
    ).to(config["device"])

    return model
