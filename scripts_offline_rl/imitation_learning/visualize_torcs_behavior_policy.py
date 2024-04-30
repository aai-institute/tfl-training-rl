from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()


from training_rl.offline_rl.behavior_policies.behavior_policy_registry import BehaviorPolicyType
from training_rl.offline_rl.custom_envs.custom_envs_registration import EnvFactory
from training_rl.offline_rl.offline_policies.offpolicy_rendering import offpolicy_rendering

env = EnvFactory.torcs.get_env()

policy = BehaviorPolicyType.torcs_expert_policy_with_noise

offpolicy_rendering(
    env_or_env_name=env,
    render_mode=None,
    behavior_policy_name=policy,
    num_frames=10000,
)

env.end()