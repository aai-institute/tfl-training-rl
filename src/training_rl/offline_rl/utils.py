import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from minari import EpisodeData
from minari.storage import get_dataset_path
from tianshou.data import Batch, ReplayBuffer
import pickle
import random

from torch.utils.data import Dataset
from IPython.display import display
from ipywidgets import widgets


def get_tianshou_root():
    try:
        # Replace "tianshou" with the name of the package/module you want to access
        with resources.path("tianshou", "..") as tianshou_root:
            return tianshou_root
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_dataset_path_d4rl(dataset_id):
    """Get the path to a dataset main directory."""
    datasets_path = os.environ.get("D4RL_DATASETS_PATH")

    if datasets_path is not None:
        file_path = os.path.join(datasets_path, "datasets", dataset_id)
    else:
        datasets_path = os.path.join(
            get_tianshou_root(), dataset_id, "offline_data", ".d4rl", "datasets"
        )
        file_path = os.path.join(datasets_path, dataset_id)

    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)


def state_action_histogram(
    state_action_count: Dict[Any, int],
    title: str = None,
    normalized=False,
    inset_pos_xy: bool = True,
    new_keys_for_state_action_count_plot: list[Any] = None,
):
    keys = list(state_action_count.keys()) if new_keys_for_state_action_count_plot is None else (
        new_keys_for_state_action_count_plot)
    values = list(state_action_count.values())
    keys_str = [str(key) if key[1] == 0 else "" for key in keys]

    if normalized:
        total_sum = sum(values)
        values = [value / total_sum for value in values]

    x_positions = np.arange(len(keys_str))

    plt.figure(figsize=(10, 5))

    plt.bar(x_positions, values, align="center", edgecolor="k")
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin as needed

    plt.xticks(x_positions, keys_str, rotation="vertical", fontsize=6)
    # plt.xticks(fontsize=12)  # Change 12 to the desired font size

    plt.xlabel("(state, action)")
    plt.ylabel("freq")
    if title is None:
        plt.title("Histogram of state_action frequencies")
    else:
        plt.title(title)

    if inset_pos_xy:
        inset_text = f"state = (x,y) ~ x + y*grid_size  -  action: (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)"  # Modify this with your desired text

        plot_height = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]  # Calculate the height of the plot
        relative_distance = -0.95 * plot_height  # Adjust the relative distance (e.g., 10% of plot height)

        inset_x = 115.0  # Adjust the x-coordinate
        inset_y = min(values) - relative_distance

        plt.text(
            inset_x,
            inset_y,
            inset_text,
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    plt.show()


def extract_dimension(input: Union["gym.core.ObsType", "gym.core.ActType"]) -> int:
    if isinstance(input, gym.spaces.Discrete):
        n = input.n
    elif isinstance(input, gym.spaces.Box):
        n = input.shape[0]
    else:
        raise ValueError(
            "So far only observations or actions that are discrete or one-dim boxes are allowed"
        )
    return n


def integer_to_one_hot(integer_value, n):
    if integer_value < 0 or integer_value > n:
        raise ValueError("Integer value is out of range [0, n]")

    one_hot_vector = np.zeros(n + 1)
    one_hot_vector[integer_value] = 1
    return one_hot_vector


def one_hot_to_integer(one_hot_vector):
    if not isinstance(one_hot_vector, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if len(one_hot_vector.shape) != 1:
        raise ValueError("Input must be a 1-dimensional array")

    return np.argmax(one_hot_vector)


def compare_state_action_histograms(
    state_action_count_1: Dict[Any, int],
    state_action_count_2: Dict[Any, int],
    title: str = None,
    normalized: bool = True,
    colors=("b", "r"),
):
    keys_1 = list(state_action_count_1.keys())
    values_1 = list(state_action_count_1.values())
    keys_str_1 = [str(key) if key[1] == 0 else "" for key in keys_1]

    keys_2 = list(state_action_count_2.keys())
    values_2 = list(state_action_count_2.values())
    keys_str_2 = [str(key) if key[1] == 0 else "" for key in keys_2]

    if normalized:
        total_sum_1 = sum(values_1)
        total_sum_2 = sum(values_2)
        values_1 = [value / total_sum_1 for value in values_1]
        values_2 = [value / total_sum_2 for value in values_2]

    x_positions_1 = np.arange(len(keys_str_1))
    x_positions_2 = np.arange(len(keys_str_2))

    plt.figure(figsize=(10, 5))

    plt.bar(
        x_positions_1,
        values_1,
        align="center",
        edgecolor="k",
        color=colors[0],
        label="Histogram 1",
        alpha=0.5,
    )
    plt.bar(
        x_positions_2,
        values_2,
        align="center",
        edgecolor="k",
        color=colors[1],
        label="Histogram 2",
        alpha=0.5,
    )
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin as needed

    plt.xticks(x_positions_1, keys_str_1, rotation="vertical", fontsize=6)

    plt.xlabel("(state, action)")
    plt.ylabel("freq")
    if title is None:
        plt.title("Comparison of State-Action Histograms")
    else:
        plt.title(title)

    plt.legend()
    plt.show()


def get_q_value_matrix(env, policy, gamma=0.99):
    state_shape = extract_dimension(env.observation_space)
    q_value_matrix = {(int1, "visited"): [0, "No"] for int1 in range(state_shape + 1)}
    visitation_mask = {key: False for key, _ in q_value_matrix.items()}

    policy_trajectory_reward = 0.0
    for i in range(1):
        done = False
        truncated = False
        state, _ = env.reset()
        t = 0
        while not (done or truncated):
            # For BCQ
            tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
            policy_output = policy(tensor_state)

            if isinstance(env.action_space, gym.spaces.Box):
                if isinstance(policy_output.act, torch.Tensor):
                    action = int(torch.argmax(policy_output.act[0]))
                elif isinstance(policy_output.act, np.ndarray):
                    action = int(np.argmax(policy_output.act[0]))
                else:
                    raise ValueError("....")
            else:
                action = int(policy_output.act[0])

            # action = int(policy_output.act[0])
            state_id = one_hot_to_integer(state)
            visitation_mask[(state_id, "visited")] = True
            next_state, reward, done, truncated, info = env.step(action)
            print(reward)
            state = next_state
            policy_trajectory_reward += reward * (gamma**t)
            t += 1

    for state_id in range(state_shape):
        ### Compute Q_values
        state = integer_to_one_hot(state_id, state_shape - 1)
        tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
        policy_output = policy(tensor_state)

        # for dqn
        # logits = policy_output.logits
        # q_values = policy.compute_q_value(logits)
        # q_values =

        # for continuous bcq
        # q_values = policy.critic2(state.reshape(1, state_shape), policy_output.act)
        # q_values = float(torch.max(policy_output.logits[0]))
        q_values = float(torch.max(policy_output.q_value[0]))
        # q_values = policy.model(state.reshape(1, state_shape))

        q_value_matrix[(state_id, "visited")][0] = q_values
        if visitation_mask[(state_id, "visited")]:
            q_value_matrix[(state_id, "visited")][1] = "Yes"

    return q_value_matrix, policy_trajectory_reward


def delete_minari_data_if_exists(file_name: str, override_dataset=True):
    local_datasets = minari.list_local_datasets()
    data_set_minari_paths = get_dataset_path("")
    custom_local_datasets = os.listdir(data_set_minari_paths)
    data_set_expert_task_path = os.path.join(data_set_minari_paths, file_name)

    if override_dataset:
        if (data_set_expert_task_path in local_datasets) or (file_name in custom_local_datasets):
            minari.delete_dataset(file_name)
        # os.makedirs(data_set_expert_task_path)
    else:
        raise FileExistsError(
            f"A dataset with that name already exists in {data_set_expert_task_path}. "
            f"Please delete it or turn 'OVERRIDE_DATA_SET' to True."
        )


def get_max_episode_steps_env(env: gym.Env) -> int:
    current_env = env
    while current_env:
        if hasattr(current_env, "spec") and current_env.spec is not None:
            max_episode_steps = current_env.spec.max_episode_steps
            if max_episode_steps is not None:
                return max_episode_steps
            else:
                raise ValueError(f"The environment doesn't have max_episode_steps.")
        if hasattr(current_env, "env"):
            current_env = current_env.env
        else:
            break


def change_max_episode_steps_env(env: gym.Env, new_max_episode_steps: int):
    current_env = env
    while current_env:
        if hasattr(current_env, "spec") and current_env.spec is not None:
            current_env.spec.max_episode_steps = new_max_episode_steps
            break
        if hasattr(current_env, "env"):
            current_env = current_env.env
        else:
            break


def _episode_data_lengths(episode: EpisodeData):
    obs = (
        episode.observations["observation"]
        if isinstance(episode.observations, Dict)
        else episode.observations
    )
    lens_data = [
        obs[1:].shape[0],
        episode.actions.shape[0],
        episode.rewards.shape[0],
        episode.truncations.shape[0],
        episode.terminations.shape[0],
    ]

    return min(lens_data)


def load_buffer_minari(expert_data_task: str) -> ReplayBuffer:
    data_set_minari_paths = get_dataset_path("")
    data_set_minari_paths = os.path.join(data_set_minari_paths, "")
    #    data_set_minari_paths = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
    data_set_expert_task_path = os.path.join(data_set_minari_paths, expert_data_task)

    if not os.path.exists(data_set_expert_task_path):
        minari.download_dataset(expert_data_task)

    dataset = minari.load_dataset(expert_data_task)

    print(f"Dataset {data_set_expert_task_path} downloaded. number of episodes: {len(dataset)}")

    observations_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []
    truncations_list = []
    next_observations_list = []

    for i, episode in enumerate(dataset):
        # For some data the len of the episode len data (observations, actions, etc.) is not the same
        common_len = _episode_data_lengths(episode)

        obs = (
            episode.observations["observation"]
            if isinstance(episode.observations, Dict)
            else episode.observations
        )

        next_observations_list.append(obs[1:][0:common_len])
        observations_list.append(obs[:-1][0:common_len])
        terminals_list.append(episode.terminations[0:common_len])
        truncations_list.append(episode.truncations[0:common_len])
        rewards_list.append(episode.rewards[0:common_len])
        actions_list.append(episode.actions[0:common_len])

    observations = np.concatenate(observations_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    terminals = np.concatenate(terminals_list, axis=0)
    next_observations = np.concatenate(next_observations_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    # truncations = np.concatenate(truncations_list, axis=0)

    replay_buffer = ReplayBuffer.from_data(
        obs=observations,
        act=actions,
        rew=rewards,
        done=terminals,
        obs_next=next_observations,
        terminated=terminals,
        truncated=np.zeros(len(terminals)),
    )
    return replay_buffer


def check_minari_files_exist(file_paths: Union[str, List[str]]) -> None:
    data_set_minari_paths = get_dataset_path("")
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        if not os.path.exists(os.path.join(data_set_minari_paths, file_path)):
            raise FileNotFoundError(f"File not found: {file_path}")


# ToDo: Improve this part of the code
def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len
        self.rtg_scale = rtg_scale

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si: si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                            dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                 torch.zeros(([padding_len] + list(actions.shape[1:])),
                                             dtype=actions.dtype)],
                                dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                       torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                   dtype=returns_to_go.dtype)],
                                      dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return timesteps, states, actions, returns_to_go, traj_mask


def widget_list(list_names: List[str], description: str = ''):
    widget_dropdown = widgets.Dropdown(options=list_names, value=list_names[0], description=description)
    display(widget_dropdown)
    return widget_dropdown


