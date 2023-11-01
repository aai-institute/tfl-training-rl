from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    RenderMode
from training_rl.offline_rl.load_env_variables import load_env_variables
from training_rl.offline_rl.offline_policies.policy_registry import PolicyName
from training_rl.offline_rl.offline_trainings.offline_training import \
    offline_training
from training_rl.offline_rl.offline_trainings.policy_config_data_class import \
    TrainedPolicyConfig

load_env_variables()

NAME_EXPERT_DATA = "Grid_2D_8x8_discrete-data_vertical_object_8x8_start_0_0_target_0_7-v0"
# "relocate-cloned-v1"
# "Ant-v2-data-v0"
POLICY_NAME = PolicyName.imitation_learning

NUM_EPOCHS = 400
BATCH_SIZE = 64
UPDATE_PER_EPOCH = 100

NUMBER_TEST_ENVS = 1
EXPLORATION_NOISE = True
SEED = None  # 1626


offline_policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_NAME,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu",
)

offline_training(
    offline_policy_config=offline_policy_config,
    num_epochs=NUM_EPOCHS,
    number_test_envs=NUMBER_TEST_ENVS,
    update_per_epoch=UPDATE_PER_EPOCH,
    restore_training=False,
    batch_size=BATCH_SIZE,
)
