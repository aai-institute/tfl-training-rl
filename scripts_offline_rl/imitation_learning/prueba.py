from training_rl.offline_rl.load_env_variables import load_env_variables
load_env_variables()
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
     EnvFactory


env = EnvFactory.torcs.get_env()

obs, _ = env.reset()

env.end()
