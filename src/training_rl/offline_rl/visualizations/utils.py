import functools
import os.path
import xml.etree.ElementTree as ET
from itertools import accumulate

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Tuple, Union, Any, Callable, Literal, List
from matplotlib import pyplot as plt
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import ImitationPolicy, BasePolicy
from torch import nn
from tqdm import tqdm

from training_rl.offline_rl.config import get_offline_rl_abs_path
from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
from training_rl.offline_rl.offline_policies.minimal_GPT_model import DecisionTransformer
from training_rl.offline_rl.offline_trainings.training_decision_transformer import evaluate_on_env
from training_rl.offline_rl.utils import extract_dimension, one_hot_to_integer, state_action_histogram, \
    compare_state_action_histograms

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
        BehaviorPolicyRestorationConfigFactoryRegistry


def ignore_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass  # Ignore KeyboardInterrupt

    return wrapper


def get_state_action_data_and_policy_grid_distributions(
    data: ReplayBuffer,
    env: gym.Env,
    policy: Union[nn.Module, str, None] = None,
    num_episodes: int = 1,
    logits_sampling: bool = False,
    plot: bool = True,
) -> Tuple[Dict, Dict]:
    """

    :param data: Tianshou ReplyBuffer dataset
    :param env:
    :param policy: a pytorch policy
    :param num_episodes: the number of episodes used to generate the policy state-action distribution.
    :param logits_sampling: if False the action will be provided (usually arg_max [Q(s,a)] ) otherwise the
        q-values will be sampled. Useful for imitation learning to compare the data and policy distributions.
    :return:
    """

    state_shape = extract_dimension(env.observation_space)
    action_shape = extract_dimension(env.action_space)

    state_action_count_data = {
        (int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)
    }

    for episode_elem in data:
        observation = episode_elem.obs
        action = episode_elem.act

        action_value = (
            int(action) if len(action.shape) == 0 or action.shape[0] <= 1 else np.argmax(action)
        )
        state_action_count_data[(one_hot_to_integer(observation), action_value)] += 1

    if policy is not None:
        state_action_count_policy = {
            (int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)
        }

        for i in tqdm(range(num_episodes), desc="Processing", ncols=100):
            done = False
            truncated = False
            state, _ = env.reset()
            while not (done or truncated):
                if policy != "random":
                    tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
                    policy_output = policy(tensor_state)

                    if logits_sampling is False:
                        action = (
                            policy_output.act[0]
                            if (
                                isinstance(policy_output.act[0], np.ndarray)
                                or isinstance(policy_output.act, np.ndarray)
                            )
                            else policy_output.act[0].detach().numpy()
                        )
                    else:
                        if isinstance(policy, ImitationPolicy):
                            q_values = policy_output.logits
                            categorical = torch.distributions.Categorical(logits=q_values[0])
                            action = np.array(categorical.sample())

                else:
                    action = env.action_space.sample()

                action_value = (
                    int(action)
                    if len(action.shape) == 0 or action.shape[0] <= 1
                    else np.argmax(action)
                )
                state_action_count_policy[(one_hot_to_integer(state), action_value)] += 1
                next_state, reward, done, truncated, info = env.step(action_value)
                state = next_state

    else:
        state_action_count_policy = None

    if plot:
        new_keys = [(env.to_xy(state_action[0]), state_action[1]) for state_action in
                    list(state_action_count_data.keys())]

        state_action_histogram(
            state_action_count_data,
            title="State-Action data distribution",
            new_keys_for_state_action_count_plot=new_keys
        )
        if state_action_count_policy is not None:
            state_action_histogram(
                state_action_count_policy,
                title="State-Action data distribution",
                new_keys_for_state_action_count_plot=new_keys
            )
            compare_state_action_histograms(state_action_count_data, state_action_count_policy)

    return state_action_count_data, state_action_count_policy


def snapshot_env(env: gym.Env):
    env.reset()
    env.step(0)
    rendered_data = env.render()  # Capture the frame as a NumPy array
    rendered_data = rendered_data[0].reshape(256, 256, 3)
    plt.imshow(rendered_data)  # Display the frame using matplotlib
    plt.show()  # Show the frame in a separate window


# ToDo: Clean this function and add, docs, etc.

def trajectory_cumulative_rewards_plot(
        env: gym.Env,
        model: DecisionTransformer,
        initial_R_0: list[float],
        trajectories_data: Dict["str", np.ndarray],
        eval_rtg_scale: float = 1000,
        num_episodes: int = 1,
        device: Literal["cpu", "cuda"] = "cpu",
        context_len: int = 20
):
    def find_closest_number_with_index(numbers, n_0):
        closest_number = min(numbers, key=lambda x: abs(x - n_0))
        closest_index = numbers.index(closest_number)
        return closest_number, closest_index

    cumulative_rewards_per_episode = []
    trajectory_length_per_episode = []
    for trajectory in trajectories_data:
        cumulative_rewards_per_episode.append(np.sum(trajectory['rewards']))
        trajectory_length_per_episode.append(len(trajectory['observations']))

    _, closest_index = find_closest_number_with_index(cumulative_rewards_per_episode, initial_R_0)

    selected_trajectory = trajectories_data[closest_index]
    selected_trajectory_length = trajectory_length_per_episode[closest_index]

    cumulative_rewards_in_time = list(accumulate(selected_trajectory["rewards"]))
    cumulative_rewards_in_time = cumulative_rewards_in_time[::-1]

    results = evaluate_on_env(
        model,
        device,
        context_len,
        env,
        initial_R_0,
        eval_rtg_scale,
        num_episodes,
        selected_trajectory_length,
        render=False,
    )

    cumulative_rewards_per_episode_inference = results['eval/rtg']

    average_cumulative_rewards_inference = np.mean(cumulative_rewards_per_episode_inference, axis=0)

    plt.plot(range(len(cumulative_rewards_in_time)), cumulative_rewards_in_time, color='blue',
             label='Average Cumulative Rewards from data')
    plt.plot(range(len(average_cumulative_rewards_inference)), average_cumulative_rewards_inference, color='red',
             label='Average Cumulative Reward Inference')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time Steps')
    plt.ylabel('Rewards to go')
    plt.title(f'Comparison of Rewards to go for initial R_0:{initial_R_0}')
    plt.show()


@ignore_keyboard_interrupt
def policy_rollout_torcs_env(
        driver_policy: BehaviorPolicyType | BasePolicy | nn.Module | Callable,
        advisor_policy: BehaviorPolicyType | BasePolicy | nn.Module | Callable | None = None,
        env_collected_quantities: str | List[str] | None = None,
        num_steps: int = 1000,
) -> Dict[str, List]:
    """
    It creates a rollout using the driver policy and collects the 'observation_name' from the environment.
    It returns a dictionary containing the actions of the driver policy and the observations collected.
    If the advisor policy is not None, it also provides the actions that should have been taken by the advisor
    policy for each step in the rollout. If env_collected_quantities are not None, it also collects those
    quantities for every step in the rollout.

    :param driver_policy:
    :param advisor_policy:
    :param env_collected_quantities:
    :param num_steps:
    """

    def get_policy(policy: BehaviorPolicyType | BasePolicy | nn.Module | Callable):
        if isinstance(policy, BehaviorPolicyType):
            return BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[policy]
        else:
            return policy

    def get_action_from_policy(obs: np.ndarray, raw_obs:Any, policy: Callable | BasePolicy) -> np.ndarray:
        if isinstance(policy, BasePolicy):
            action = policy(Batch({"obs": obs.reshape(1, 19), "info": {}})).act
            if isinstance(action, torch.Tensor):
                action = action.detach().numpy()
            return action
        elif isinstance(policy, nn.Module):
            action = policy(torch.Tensor(obs))
            if isinstance(action, tuple):
                action = action[0]
            return action.detach().numpy()
        elif isinstance(policy, Callable):
            return policy(raw_obs)

    driver = get_policy(driver_policy)
    advisor = None if advisor_policy is None else get_policy(advisor_policy)

    list_actions_advisor = []
    list_actions_driver = []
    list_observations = []

    collected_quantities = {}
    if env_collected_quantities is not None:
        if isinstance(env_collected_quantities, str):
            env_collected_quantities = [env_collected_quantities]
        collected_quantities = {key: [] for key in env_collected_quantities}

    env = EnvFactory.torcs.get_env()
    obs, _ = env.reset()

    done = False
    for num_step in range(num_steps):
        if done:
            break

        raw_obs = env.raw_observation

        list_observations.append(obs)
        if env_collected_quantities is not None:
            for collected_quantity in env_collected_quantities:
                if collected_quantity in raw_obs:
                    collected_quantities[collected_quantity].append(raw_obs[collected_quantity].item())
                else:
                    raise ValueError(f"observation {env_collected_quantities} not found in environment")

        action = get_action_from_policy(obs, raw_obs, driver)

        action_value = action if not isinstance(action, np.ndarray) else action.item()
        list_actions_driver.append(action_value)

        if advisor_policy is not None:
            action_advisor = get_action_from_policy(obs, raw_obs, advisor)
            action_advisor = action_advisor if not isinstance(action_advisor, np.ndarray) else action_advisor.item()
            list_actions_advisor.append(action_advisor)

        obs, reward, done, truncations, info = env.step(action)

    env.end()

    output = {
        "actions_driver": list_actions_driver,
        "actions_advisor": list_actions_advisor,
        "observations": list_observations,
    }

    if env_collected_quantities is not None:
        output.update(collected_quantities)

    return output


def compare_policy_decisions_vs_expert_suggestions(
    policy_actions:List[np.ndarray],
    expert_suggestions:List[np.ndarray],
    iteration:int = 0,
):
    x = range(len(policy_actions))
    plt.plot(x, policy_actions, label='Policy actions')
    plt.plot(x, expert_suggestions, label='Expert suggested actions')
    plt.xlabel('Time steps')
    plt.ylabel("Actions")
    plt.title(f'Policy actions vs expert suggestions in Dagger iter: {iteration}')
    plt.legend()
    plt.show()


