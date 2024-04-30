import numpy as np

from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.dagger_torcs_policy import model_dagger_fit

load_env_variables()


from training_rl.offline_rl.behavior_policies.behavior_policy_registry import \
    BehaviorPolicyRestorationConfigFactoryRegistry

import os
import torch
from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
from training_rl.offline_rl.generate_custom_minari_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName, PolicyFactoryRegistry
from training_rl.offline_rl.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from training_rl.offline_rl.offline_trainings.restore_policy_model import restore_trained_offline_policy
from training_rl.offline_rl.utils import load_buffer_minari
from training_rl.offline_rl.visualizations.utils import compute_corrected_actions_from_policy_guided, \
    update_torcs_display_mode


def get_tensor_from_array(array:np.ndarray) -> torch.Tensor:
    return torch.Tensor(array)

# 1 - Restoration of BC policy (only important for initial rollout)

DATA_SET_SUBOPTIMAL_NAME = "torcs-torcs_suboptimal-v0"
POLICY_NAME_EXERCISE_III_B = "policy_bc_epoch_1.pt"
OFFLINE_POLICY_NAME = PolicyName.imitation_learning_torcs
DEVICE = "cuda"

DAGGER_NUM_STEPS = 5000
DAGGER_POLICY_NAME = "dagger_trocs.pt"
DAGGER_ITERS = 20



buffer_data = load_buffer_minari(DATA_SET_SUBOPTIMAL_NAME)
data_config = MinariDatasetConfig.load_from_file(DATA_SET_SUBOPTIMAL_NAME)

offline_policy_config = TrainedPolicyConfig(
    name_expert_data=DATA_SET_SUBOPTIMAL_NAME,
    policy_name= OFFLINE_POLICY_NAME,
    device=DEVICE
)

trained_bc_policy = restore_trained_offline_policy(offline_policy_config)
log_name = os.path.join(DATA_SET_SUBOPTIMAL_NAME, OFFLINE_POLICY_NAME)
log_path = get_trained_policy_path(log_name)
trained_bc_policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_NAME_EXERCISE_III_B), map_location="cpu"))


# 2 - initial expert correction

policy_expert = BehaviorPolicyRestorationConfigFactoryRegistry.torcs_expert_policy

initial_output = compute_corrected_actions_from_policy_guided(
    env_name=EnvFactory.torcs,
    policy_guide=trained_bc_policy,
    policy_a=policy_expert,
    #policy_b=trained_bc_policy,
    num_steps=DAGGER_NUM_STEPS,
    visualize=True
)

# 3  - create our dagger policy (we could use the previous BC one but we will create a new one).
dagger_offline_policy = PolicyFactoryRegistry.dagger_torcs()

corrected_actions = np.array(initial_output["actions_corrected_policy"])
collected_states = np.array(initial_output["collected_states"])

for dagger_iter in range(DAGGER_ITERS):

    model_dagger_fit(
        input_data=get_tensor_from_array(collected_states),
        target_data=get_tensor_from_array(corrected_actions),
        model=dagger_offline_policy
    )

    output = compute_corrected_actions_from_policy_guided(
        env_name=EnvFactory.torcs,
        policy_guide=dagger_offline_policy,
        policy_a=policy_expert,
        #policy_b=trained_bc_policy,
        num_steps=DAGGER_NUM_STEPS,
        visualize=True
    )

    corrected_actions = np.concatenate([np.array(output["actions_corrected_policy"]), corrected_actions], axis=0)
    collected_states = np.concatenate([np.array(output["collected_states"]), collected_states], axis=0)

    #model_observation_all = np.concatenate([model_observation_all, model_observations], axis=0)


#model_dagger_fit(
#    input_data: torch.Tensor,
#    target_data: torch.Tensor,
#    model: nn.Module,
#    batch_size=128,
#    epochs=1,
#    shuffle=True,
#):



#for iter in range(DAGGER_ITER):







