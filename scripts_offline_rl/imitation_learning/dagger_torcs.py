from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()

import os
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch
from torch import nn

from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
from training_rl.offline_rl.custom_envs.gym_torcs.gym_torcs import TorcsLidarEnv
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.offline_training import offline_training
from training_rl.offline_rl.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from training_rl.offline_rl.offline_trainings.restore_policy_model import restore_trained_offline_policy
from training_rl.offline_rl.utils import load_buffer_minari



def compare_expert_vs_model_actions(array1, array2, label1="", label2="", title=""):
    if array1.shape != array2.shape:
        raise ValueError(f"Arrays have different shapes")

    x_values = np.arange(array1.size)
    plt.plot(x_values, array1, label=label1)
    plt.plot(x_values, array2, label=label2)

    # Highlight differing elements
    differing_indices = np.where(array1 != array2)[0]
    plt.scatter(differing_indices, array1[differing_indices], color='red', label='Different Element')
    plt.scatter(differing_indices, array2[differing_indices], color='red')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()


def rollout_and_collect_states_and_actions(
        env: TorcsLidarEnv,
        policy: nn.Module | Callable[[np.ndarray], np.ndarray],
        num_steps: int = 1000,
        collect_states=True,
        collect_actions=True,
        ) -> tuple[list["state"], list[np.ndarray]]:

    states = []
    actions = []
    state, _ = env.reset(relaunch=True)
    states.append(state)

    # ToDo: progress bar here
    print('Collecting data...')
    for iter in range(num_steps):

        if iter == 0:
            act = np.array([0.0])
        else:
            if isinstance(policy, nn.Module):
                input_tensor_obs = torch.tensor(np.array([state]), dtype=torch.float32)
                input_tensor_obs_batch = Batch({"obs": input_tensor_obs, "info": {}})
                policy_output = policy(input_tensor_obs_batch)
                act = policy_output.act[0].detach().numpy()
            elif isinstance(policy, Callable):
                act = policy(state)
            else:
                raise ValueError(f"Unrecognized mode_or_expert: {policy}")

        state, reward, done, _, _ = env.step(act)
        if collect_states:
            states.append(state)
        if collect_actions:
            actions.append(act)

        if done:
            break

    return states[:-1], actions  # the last observation doesn't have an action associated


if __name__ == '__main__':

    # ToDo: This MUST BE AN EXERCISE.
    # ToDo: Huge code duplication with offline_training_evaluation_torcs.py

    NUM_EPOCHS = 1
    BATCH_SIZE = 128
    NUMBER_TEST_ENVS = 1
    EXPLORATION_NOISE = True
    SEED = None  # 1626
    PERCENTAGE_DATA_PER_EPOCH = 0.8
    POLICY_NAME = PolicyName.imitation_learning_torcs
    DATA_SET_NAME = "torcs-torcs_suboptimal-v0"
    NUMBER_COLLECTED_STEPS = 10000
    POLICY_FILE = "policy_bc_epoch_3.pt"

    NUM_DAGGER_ITERATIONS = 10

    # data = minari.load_dataset(DATA_SET_NAME)
    buffer_data = load_buffer_minari(DATA_SET_NAME)

    data_config = MinariDatasetConfig.load_from_file(DATA_SET_NAME)

    offline_policy_config = TrainedPolicyConfig(
        name_expert_data=DATA_SET_NAME,
        policy_name=POLICY_NAME,
        device="cpu"
    )




    # restore a policy with the same configuration as the one we trained.
    policy = restore_trained_offline_policy(offline_policy_config)
    # load the weights
    name_expert_data = offline_policy_config.name_expert_data
    log_name = os.path.join(name_expert_data, POLICY_NAME)
    log_path = get_trained_policy_path(log_name)
    policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_FILE), map_location="cpu"))
    

    # 1 - Get initial states and obs from data
    collected_observations = buffer_data.obs
    expert_collected_actions = buffer_data.act
    # Compare with BC policy action predictions
    policy_collected_actions = policy(Batch({"obs": buffer_data.obs, "info": {}})).act.detach()
    compare_expert_vs_model_actions(expert_collected_actions, policy_collected_actions, title="BC phase")

    # Dagger algorithm

    # ToDo: add a progress bar
    for itr in range(NUM_DAGGER_ITERATIONS):
        env = EnvFactory[data_config.env_name].get_env()

        # 2 - Use trained policy to collect new observations
        policy_observations, policy_actions = rollout_and_collect_states_and_actions(
            env=env,
            policy=policy,
            num_steps=NUMBER_COLLECTED_STEPS,
        )

        # 3 - Correct policy actions with expert knowledge
        corrected_policy_actions_by_expert = policy(Batch({"obs": policy_observations, "info": {}})).act.detach().numpy()

        collected_observations = np.concatenate([collected_observations, policy_observations], axis=0)
        expert_collected_actions = np.concatenate([expert_collected_actions,
                                                   corrected_policy_actions_by_expert], axis=0)
        policy_collected_actions = np.concatenate([policy_collected_actions, policy_actions], axis=0)

        compare_expert_vs_model_actions(
            np.array(policy_actions, dtype=np.float32),
            corrected_policy_actions_by_expert,
            title="Aggregation phase"
        )

        collected_observations_tensors = torch.tensor(collected_observations, dtype=torch.float32)
        expert_collected_actions_tensors = torch.tensor(expert_collected_actions, dtype=torch.float32)

        offline_training(
            offline_policy_config=offline_policy_config,
            num_epochs=NUM_EPOCHS,
            number_test_envs=NUMBER_TEST_ENVS,
            step_per_epoch=PERCENTAGE_DATA_PER_EPOCH * len(buffer_data),
            restore_training=True,
            batch_size=BATCH_SIZE,
            policy_name="dagger_policy.ph"
        )

        env.end()

