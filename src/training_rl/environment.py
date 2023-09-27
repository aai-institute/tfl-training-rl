import math

import numpy as np
from gymnasium import Env
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker, TimeLimit
from gymnasium.wrappers.render_collection import RenderCollection

__all__ = ["create_inverted_pendulum_environment", "InvertedPendulumEnvWithCutoffAngle"]


class InvertedPendulumEnvWithCutoffAngle(InvertedPendulumEnv):
    """Modified version of InvertedPendulumEnv that allows setting a different cutoff angle."""

    def __init__(self, *args, **kwargs) -> None:
        self.cutoff_angle = kwargs.pop("cutoff_angle", math.radians(45))
        super().__init__(*args, **kwargs)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > self.cutoff_angle))
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}


def create_inverted_pendulum_environment(
    render_mode: str = "rgb_array",
    max_steps: int = 100,
    cutoff_angle: float = 0.8,
) -> Env:
    """Creates instance of InvertedPendulumEnvWithCutoffAngle with some wrappers
    to ensure correctness, limit the number of steps and store rendered frames.
    """
    env = InvertedPendulumEnvWithCutoffAngle(cutoff_angle=cutoff_angle, render_mode=render_mode)
    env = PassiveEnvChecker(env)
    env = OrderEnforcing(env)
    env = TimeLimit(env, max_steps)
    return RenderCollection(env)
