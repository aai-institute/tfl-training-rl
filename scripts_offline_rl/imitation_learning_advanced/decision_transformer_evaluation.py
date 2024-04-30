from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import pickle

import numpy as np
from matplotlib import pyplot as plt
import gym

import torch
from training_rl.offline_rl.offline_policies.decision_transformer_policy import get_decision_transformer_default_config, \
    create_decision_transformer_policy_from_dict
from training_rl.offline_rl.offline_policies.minimal_GPT_model import evaluate_on_env


if __name__ == "__main__":

    PATH_TO_MODEL = \
        "../../src/training_rl/offline_rl/data/decision_transformers/models/model_1000_ep_1000_steps_halfcheetah/model_d4rl_halfcheetah_medium_v0_April_24_v2.pt"
    ENV_NAME = "HalfCheetah-v3"  # "Walker2d-v3"
    DATASET_PATH = "../../src/training_rl/offline_rl/data/decision_transformers/d4rl_data/halfcheetah-medium-v0.pkl"


    #PATH_TO_MODEL = \
    #    "../../src/training_rl/offline_rl/data/decision_transformers/models/model_d4rl_walker2d-medium-v1_April_24_v0.pt"
    #ENV_NAME = "Walker2d-v3"
    #DATASET_PATH = "../../src/training_rl/offline_rl/data/decision_transformers/d4rl_data/walker2d-medium-v1.pkl"

    EVAL_RTG_TARGET = 500
    EVAL_RTG_SCALE = 1000  # normalize returns to goq
    NUM_EPISODES = 20  # num of evaluation episodes
    NUM_STEPS_PER_EPISODE = 4096  # max len of one episode --> max dimension of the embedding time-steps in the DT
    RENDER_MODE = False
    PLOT_CUMULATIVE_REWARDS = True

    env = gym.make(ENV_NAME, render_mode='rgb_array' if RENDER_MODE else None)

    decision_transformer_config = get_decision_transformer_default_config()
    decision_transformer_config["device"] = "cpu"
    device = decision_transformer_config["device"]
    context_len = decision_transformer_config["context_len"]

    model = create_decision_transformer_policy_from_dict(
        config=decision_transformer_config,
        action_space=env.action_space,
        observation_space=env.observation_space
    )

    model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
    print("Policy loaded from: ", PATH_TO_MODEL)

    results = evaluate_on_env(
        model,
        device,
        context_len,
        env,
        EVAL_RTG_TARGET,
        EVAL_RTG_SCALE,
        NUM_EPISODES,
        NUM_STEPS_PER_EPISODE,
        render=RENDER_MODE
    )

    if PLOT_CUMULATIVE_REWARDS:
    
        with open(DATASET_PATH, 'rb') as f:
            trajectories = pickle.load(f)

        cumulative_rewards_per_episode = []
        for trajectory in trajectories:
            cumulative_rewards_per_episode.append(np.sum(trajectory['rewards']))

        plt.figure(figsize=(8, 6))
        plt.hist(cumulative_rewards_per_episode, bins=10, edgecolor='black')
        plt.xlabel('Average Cumulative Rewards')
        plt.ylabel('Frequency')
        plt.title('Histogram of Average Cumulative Rewards per Episode')
        plt.grid(True)
        
        plt.show()

    
        cumulative_rewards = results['eval/cumulative_reward_per_episode']
        average_rewards = np.zeros(NUM_STEPS_PER_EPISODE)  # Initialize an array to store average rewards
        episode_counts = np.zeros(NUM_STEPS_PER_EPISODE, dtype=int)  # Initialize an array to count episodes contributing to each step

        # Calculate cumulative rewards and episode counts for each step
        for episode_rewards in cumulative_rewards:
            for step, cumulative_reward in enumerate(episode_rewards):
                average_rewards[step] += cumulative_reward
                episode_counts[step] += 1

        average_rewards /= np.maximum(episode_counts, 1)  # Avoid division by zero

        # Plot the average cumulative reward over time
        plt.figure(figsize=(8, 6))
        plt.plot(range(NUM_STEPS_PER_EPISODE), average_rewards, label='Average Cumulative Reward')
        plt.xlabel('Time Step')
        plt.ylabel('Average Cumulative Reward')
        plt.title('Average Cumulative Reward Over Time (Variable Episode Lengths)')
        plt.legend()
        plt.grid(True)
        plt.show()
        

