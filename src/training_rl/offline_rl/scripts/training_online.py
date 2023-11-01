from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    RenderMode
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.policy_registry import (
    PolicyName, PolicyType)
from training_rl.offline_rl.offline_trainings.policy_config_data_class import \
    TrainedPolicyConfig
from training_rl.offline_rl.online_trainings.online_training import \
    online_training

load_env_variables()

NAME_EXPERT_DATA = (
    # "Grid_2D_8x8_discrete-data_vertical_object_8x8_start_0_0_target_0_7-v0"
    "Grid_2D_8x8_continuous-data_vertical_object_8x8_start_0_0_target_0_7-v0"
)
# "relocate-cloned-v1"
# "Grid_2D_8x8_discrete-combined_data_set-V0"
POLICY_NAME = PolicyName.ppo
POLICY_TYPE = PolicyType.onpolicy

NUM_EPOCHS = 1
BATCH_SIZE = 64
STEP_PER_EPOCH = 100000
STEP_PER_COLLECT = 10

NUMBER_TRAINING_ENVS = 10
NUMBER_TEST_ENVS = 5
EXPLORATION_NOISE = True
SEED = None  # 1626


policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_NAME,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu",
)


if POLICY_NAME == PolicyName.ppo:
    policy_config.policy_config["lr_decay"] = {
        "step_per_epoch": STEP_PER_EPOCH,
        "step_per_collect": STEP_PER_COLLECT,
        "epoch": NUM_EPOCHS,
    }


online_training(
    trained_policy_config=policy_config,
    policy_type=POLICY_TYPE,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    number_train_envs=NUMBER_TRAINING_ENVS,
    step_per_epoch=STEP_PER_EPOCH,
    step_per_collect=STEP_PER_COLLECT,
    number_test_envs=NUMBER_TEST_ENVS,
    restore_training=False,
)
