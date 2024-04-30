import os.path
from typing import Callable, Optional

import minari
import numpy as np
import matplotlib.pyplot as plt
import torch
from minari import DataCollectorV0, StepDataCallback
from minari.dataset import minari_dataset
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
    BehaviorPolicyRestorationConfigFactoryRegistry
from training_rl.offline_rl.custom_envs.custom_envs_registration import register_grid_envs
from training_rl.offline_rl.custom_envs.gym_torcs.gym_torcs import TorcsEnv
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import \
    create_minari_collector_env_wrapper, MinariDatasetConfig, create_minari_config
from training_rl.offline_rl.utils import delete_minari_data_if_exists

register_grid_envs()


class DaggerPolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(DaggerPolicyModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.angle_output = nn.Linear(512, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.shared_layer(x)
        x = self.angle_output(x)
        x = self.tanh(x)
        return x


# ToDo: Duplicated code!!

# ToDo Ready to Remove
def create_minari_datasets(
    env_name: str,
    dataset_name: str = "data",
    dataset_identifier: str = "",
    version_dataset: str = "v0",
    num_colected_points: int = 1000,
    behavior_policy_name: BehaviorPolicyType = BehaviorPolicyType.random,
) -> MinariDatasetConfig:
    """
    Creates a custom Minari dataset and save a MinariDatasetConfig metadata to file (see /data/offline_data).

    :param env_name:
    :param dataset_name:
    :param dataset_identifier:
    :param version_dataset:
    :param num_colected_points:
    :param behavior_policy_name: One of our registered behavioral policies (see behavior_policy_registry.py).
    :param env_2d_grid_initial_config: If the environment is of type Custom2DGridEnv, its initial config
    :return:
    :rtype:
    """
    dataset_config = create_minari_config(
        env_name=env_name,
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        version_dataset=version_dataset,
        num_steps=num_colected_points,
        behavior_policy_name=behavior_policy_name
    )

    delete_minari_data_if_exists(dataset_config.data_set_name, override_dataset=True)
    env = create_minari_collector_env_wrapper(
        dataset_config.env_name, dataset_config.initial_config_2d_grid_env
    )
    state, _ = env.reset()

    num_steps = 0
    for _ in range(dataset_config.num_steps):
        behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[
            dataset_config.behavior_policy
        ]

        if dataset_config.behavior_policy == "random":
            action = env.action_space.sample()
        elif isinstance(env.unwrapped, TorcsEnv):
            action = np.array(behavior_policy(state), dtype=np.float32)
        else:
            action = behavior_policy(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_steps += 1

        if done or time_out:
            state, _ = env.reset()
            num_steps = 0
        else:
            state = next_state

    if isinstance(env.unwrapped, TorcsEnv):
        env.end()

    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_config.data_set_name, collector_env=env
    )
    dataset_config.save_to_file()

    return dataset_config

# ToDo Ready to Remove
def get_teacher_action(state: "state") -> np.ndarray:
    """
    suboptimal expert policy: It is not really an expert policy as it has a hard time to do a good job when the car
        is out of the road.
    """
    steer = state["angle"] * 10 / np.pi
    steer -= state["trackPos"] * 0.10
    return np.array([steer])


def observation_preprocessing(state: "state") -> np.ndarray:
    """
     Observation preprocessing: We only consider lidar sensors
    """
    return np.array([lidar for lidar in state["track"]])


def rollout_and_collect_states_and_actions(
        env: TorcsEnv,
        mode_or_expert: DaggerPolicyModel | Callable[[np.ndarray], np.ndarray] = get_teacher_action,
        num_steps: int = 1000,
        collect_states=True,
        collect_actions=True,
        clip_range: Optional[tuple[float, float]] = None,
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
            if isinstance(mode_or_expert, DaggerPolicyModel):
                input_tensor_obs = torch.tensor(np.array([observation_preprocessing(state)]), dtype=torch.float32)
                act = mode_or_expert(input_tensor_obs).detach().cpu().numpy()
                if clip_range:
                    act = np.clip(act[0], clip_range[0], clip_range[1])
            elif isinstance(mode_or_expert, Callable):
                act = mode_or_expert(state)
            else:
                raise ValueError(f"Unrecognized mode_or_expert: {mode_or_expert}")

        state, reward, done, _, _ = env.step(act)
        if collect_states:
            states.append(state)
        if collect_actions:
            actions.append(act)

        if done:
            break

    return states[:-1], actions  # the last observation doesn't have an action associated


def get_expert_actions_from_observations(observations: list[np.ndarray]):
    action_list = []
    for obs in observations:
        action_list.append(get_teacher_action(obs))
    return action_list


def compare_expert_vs_model_actions(array1, array2, label1="", label2="", title=""):
    if array1.shape != array2.shape:
        print("Arrays have different shapes")
        return

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


def model_fit(
    model: DaggerPolicyModel,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    batch_size = 128,
    epochs=1,
    shuffle=True,
    callbacks=None,
):
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs_batch, targets_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}')


if __name__ == '__main__':

    num_lidars = 19  # Lidar sensors
    initial_steps = 10
    # episode steps in Behavioral Cloning phase
    dagger_itr = 1
    steps = 20  # episode steps during dagger iteration

    path_to_model = '/home/ivan/Documents/GIT_PROJECTS/imitation-dagger/trained_model.h5'
    train_from_scratch = True

    batch_size = 64
    nb_epoch = 100
    #tensorboard_callback = TensorBoard(log_dir='/home/ivan/Documents/GIT_PROJECTS/imitation-dagger/logs',
    #                                   histogram_freq=1)

    CLIP_RANGE = (-0.4, 0.4)

    if not train_from_scratch:
        if not os.path.exists(path_to_model):
            raise ValueError("The path for your model does not exist")

    # ToDo:
    #  1 . Add Minari data collection
    #  2. Load in tianshou and do BC or CQL, etc.

    #minari_dataset = create_minari_datasets(
    #    env_name="torcs",
    #    behavior_policy_name=BehaviorPolicyType.torcs_expert_policy,
    #    num_colected_points=initial_steps
    #)

    #data = minari.load_dataset(minari_dataset.data_set_name)




    env = TorcsEnv(throttle=False)  # No vision for state vector

    initial_state_list, initial_action_list = rollout_and_collect_states_and_actions(
        env=env,
        mode_or_expert=get_teacher_action,
        num_steps=initial_steps
    )
    model = DaggerPolicyModel(input_dim=num_lidars)
    env.end()

    initial_observation_list = [observation_preprocessing(state) for state in initial_state_list]
    # Convert lists to numpy arrays
    model_observation_all = np.array(initial_observation_list)
    expert_actions_all = np.array(initial_action_list)

    # Compile the model
    model_observation_all_tensors = torch.tensor(model_observation_all, dtype=torch.float32)
    expert_actions_all_tensors = torch.tensor(expert_actions_all, dtype=torch.float32)

    model_fit(
          model = model,
          input_data=model_observation_all_tensors,
          target_data=expert_actions_all_tensors,
          batch_size=batch_size,
          epochs=nb_epoch,
          shuffle=True,
          #callbacks=tensorboard_callback
        )

    torch.save(
        model,
        '/scripts_offline_rl/dagger_exercise/dagger_model.h5'
    )

    model_actions_all = model(model_observation_all_tensors).detach().cpu().numpy()
    compare_expert_vs_model_actions(expert_actions_all, model_actions_all, title="BC phase")


    # ToDo: add a progress bar
    for itr in range(dagger_itr):
        env = TorcsEnv(vision=False, throttle=False)  # No vision for state vector

        model_states, model_actions = rollout_and_collect_states_and_actions(
            env=env,
            mode_or_expert=model,
            num_steps=steps,
            clip_range=CLIP_RANGE,
        )

        env.end()
        print('Episode done ', itr)

        expert_actions = np.array(get_expert_actions_from_observations(model_states))

        model_observations = [observation_preprocessing(model_state) for model_state in model_states]
        model_observation_all = np.concatenate([model_observation_all, model_observations], axis=0)
        expert_actions_all = np.concatenate([expert_actions_all, expert_actions], axis=0)
        model_actions_all = np.concatenate([model_actions_all, model_actions], axis=0)

        compare_expert_vs_model_actions(expert_actions, np.array(model_actions), title="Aggregation phase")

        # Retrain the model with expert data
        # ToDo: Change this
        model_observation_all_tensors = torch.tensor(model_observation_all, dtype=torch.float32)
        expert_actions_all_tensors = torch.tensor(expert_actions_all, dtype=torch.float32)
        model_fit(
            model,
            model_observation_all_tensors,
            expert_actions_all_tensors,
            batch_size=batch_size,
            epochs=nb_epoch,
            shuffle=True,
            #callbacks=tensorboard_callback
            )

        torch.save(
            model,
            '/scripts_offline_rl/dagger_exercise/dagger_model.h5'
        )
