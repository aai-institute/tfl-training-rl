from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
from gymnasium import Wrapper

from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.simple_grid import \
    Custom2DGridEnv


@dataclass
class Grid2DInitialConfig:
    obstacles: ObstacleTypes = None
    initial_state: Tuple = None
    target_state: Tuple = None


def is_instance_of_Custom2DGridEnv(env):
    while isinstance(env, gym.Env):
        if isinstance(env, Custom2DGridEnv):
            return True
        if hasattr(env, "env"):
            env = env.env
        else:
            break
    return False


class InitialConfigCustom2DGridEnvWrapper(Wrapper, gym.utils.RecordConstructorArgs):
    """
    This wrapper is only used to change initial conditions for the Custom2DGridEnv (e.g.  different obstacles, or
    initial/target states) and it hasn't any effect on other environments. Initial conditions cannot be changed
    in __init__ as this will produce two different environments from the point of view of Minari (teo different Spec's),
    as the metadata will be different (you could have the same effect passing a kwargs to reset)
    """

    def __init__(self, env, env_config: Grid2DInitialConfig = None):
        super().__init__(env)
        if is_instance_of_Custom2DGridEnv(self.env):
            if env_config is not None:
                obstacles = env_config.obstacles
                if obstacles is not None:
                    self.env.set_new_obstacle_map(obstacles.value)
                initial_state = env_config.initial_state
                if initial_state is not None:
                    self.env.set_starting_point(initial_state)
                target_state = env_config.target_state
                if target_state is not None:
                    self.env.set_goal_point(target_state)
