import os.path
import xml.etree.ElementTree as ET
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Tuple, Union, Any, Callable, Literal
from matplotlib import pyplot as plt
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import ImitationPolicy, BasePolicy
from torch import nn
from tqdm import tqdm

from training_rl.offline_rl.config import get_offline_rl_abs_path
from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
from training_rl.offline_rl.utils import extract_dimension, one_hot_to_integer

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
        BehaviorPolicyRestorationConfigFactoryRegistry



def get_state_action_data_and_policy_grid_distributions(
    data: ReplayBuffer,
    env: gym.Env,
    policy: Union[nn.Module, str, None] = None,
    num_episodes: int = 1,
    logits_sampling: bool = False,
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

    return state_action_count_data, state_action_count_policy


def snapshot_env(env: gym.Env):
    env.reset()
    env.step(0)
    rendered_data = env.render()  # Capture the frame as a NumPy array
    rendered_data = rendered_data[0].reshape(256, 256, 3)
    plt.imshow(rendered_data)  # Display the frame using matplotlib
    plt.show()  # Show the frame in a separate window


def compute_corrected_actions_from_policy_guided(
        env_name: EnvFactory,
        policy_guide: BehaviorPolicyType | BasePolicy | nn.Module | Callable,
        policy_a: BehaviorPolicyType | BasePolicy | nn.Module | Callable,
        policy_b: BehaviorPolicyType | BasePolicy | nn.Module | Callable | None = None,
        num_steps=1000,
        visualize: bool = True,
) -> Dict:
    """
    Create an episode in 'env_name' following 'policy_guide', then compute, for each state selected by the
    'policy_guide', the actions that would have been taken if the 'policy_a'  or 'policy_b' had been used.
    This is a useful function for the DAGGER algorithm. It returns a Dict with the list of collected actions and
    the collected states.
    """

    def plot_policies_actions(lists_actions, lists_labels, plot_title):
        lens = [len(list_actions) for list_actions in lists_actions]
        if len(set(lens)) != 1:
            raise ValueError("policy_actions have different shapes")

        if lists_labels is None:
            lists_labels = ("",) * len(lists_actions[0])

        x_values = np.arange(len(lists_actions[0]))
        for actions, label in zip(lists_actions, lists_labels):
            plt.scatter(x_values, actions, label=label)

        plt.xlabel('step')
        plt.ylabel('action')
        plt.title(plot_title)
        plt.legend()
        plt.show()

    def get_policy(policy: BehaviorPolicyType | BasePolicy):
        if isinstance(policy, BehaviorPolicyType):
            return BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[policy]
        else:
            return policy

    def get_action_from_policy(obs: np.ndarray, raw_obs:Any, policy: Callable | BasePolicy) -> np.ndarray:
        if isinstance(policy, BasePolicy):
            return policy(Batch({"obs": obs, "info": {}})).act.detach().numpy()
        elif isinstance(policy, nn.Module):
            return policy(torch.Tensor(obs)).detach().numpy()
        elif isinstance(policy, Callable):
            return policy(raw_obs)

    def get_policy_name(policy: BehaviorPolicyType | BasePolicy):
        if isinstance(policy, BehaviorPolicyType):
            return policy.name
        else:
            return type(policy).__name__

    policy_names = [get_policy_name(policy_guide), get_policy_name(policy_a)]
    if policy_b is not None:
        policy_names += [get_policy_name(policy_b)]

    policy_rollout = get_policy(policy_guide)
    policy_1 = get_policy(policy_a)
    if policy_b is not None:
        policy_2 = get_policy(policy_b)

    list_actions_policy_rollout = []
    list_actions_policy_1 = []
    list_actions_policy_2 = []
    list_obs = []

    env = EnvFactory[env_name].get_env()
    obs, _ = env.reset()
    for num_step in range(num_steps):
        raw_obs = env.raw_observation
        action = get_action_from_policy(obs, raw_obs, policy_rollout)
        list_actions_policy_rollout.append(action)
        list_actions_policy_1.append(get_action_from_policy(obs, raw_obs, policy_1))
        if policy_b is not None:
            list_actions_policy_2.append(get_action_from_policy(obs, raw_obs, policy_2))
        list_obs.append(obs)
        obs, reward, done, truncations, info = env.step(action)
    env.end()

    list_policies_actions = [list_actions_policy_rollout, list_actions_policy_1]
    if policy_b is not None:
        list_policies_actions += [list_actions_policy_2]

    labels = policy_names

    if visualize:
        title = "action comparison"
        plot_policies_actions(list_policies_actions, labels, title)

    output = {
        "actions_guided_policy": list_actions_policy_rollout,
        "actions_corrected_policy": list_actions_policy_1,
        "collected_states": list_obs,
    }

    return output


def update_torcs_display_mode(new_display_mode:Literal["results_only", "normal"] = "normal"):

    path_to_script = os.path.join(get_offline_rl_abs_path(), "..", "..", '..', 'torcs', 'BUILD',
                                  'share', 'games', 'torcs', 'config', 'raceman', 'practice.xml')


    # Load the XML file
    tree = ET.parse(path_to_script)

    root = tree.getroot()

    # Find the section with name="Practice"
    practice_section = root.find(".//section[@name='Practice']")

    # Find the attribute with name="display mode" within the Practice section
    display_mode_attr = practice_section.find("./attstr[@name='display mode']")

    if display_mode_attr is not None:
        # Update the value of the display mode attribute
        display_mode_attr.set('val', new_display_mode)

        # Save the modified XML back to the file
        tree.write(path_to_script)
        print(f"Display mode updated to '{new_display_mode}' in the XML file.")
    else:
        print("Attribute 'display mode' not found in the Practice section.")


