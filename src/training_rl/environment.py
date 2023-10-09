import importlib.resources
import math
import os
from typing import ClassVar

import numpy as np
from gymnasium import Env, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker, TimeLimit
from gymnasium.wrappers.render_collection import RenderCollection

__all__ = ["create_inverted_pendulum_environment", "create_mass_spring_damper_environment"]


ASSETS_DIR = importlib.resources.files(__package__) / "assets"


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


def create_mass_spring_damper_environment(
    render_mode: str = "rgb_array",
    max_steps: int = 100,
) -> Env:
    """Creates instance of MassSpringDamperEnv with some wrappers
    to ensure correctness, limit the number of steps and store rendered frames.
    """
    env = MassSpringDamperEnv(render_mode=render_mode)
    env = PassiveEnvChecker(env)
    env = OrderEnforcing(env)
    env = TimeLimit(env, max_steps)
    return RenderCollection(env)


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


class MassSpringDamperEnv(MujocoEnv, utils.EzPickle):
    """## Description.

    This environment is a vertical mass-spring-damper environment based on the classical control problem.

    ## Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-10 10]`, where `action` represents
    the numerical force applied to the mass (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the mass | -20          | 20           | slider                           | slide | Force (N) |

    ## Observation Space

    The state space consists of positional values of different body parts of
    the mass-spring-damper system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the mass along the negative z-axis | -0.32 | 0.32 | mass                           | slide | position (m)              |
    | 1   | linear velocity of the mass                   | -Inf | Inf | mass                           | slide | velocity (m/s)            |

    ## Starting State
    All observations start in state (0.0, 0.0).

    ## Episode End
    The episode ends when any of the following happens:

    1. Truncation: The episode duration reaches 1000 timesteps.
    2. Termination: Any of the state space values is no longer finite.

    """

    metadata: ClassVar = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    default_camera_config: ClassVar = {
        "distance": 1.0,
        "elevation": -5,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(
            low=np.array([-0.32, -np.inf]),
            high=np.array([0.32, np.inf]),
            shape=(2,),
            dtype=np.float64,
        )
        model_file = os.fspath(ASSETS_DIR / "mass_spring.xml")
        MujocoEnv.__init__(
            self,
            model_file,
            2,
            observation_space=observation_space,
            default_camera_config=self.default_camera_config,
            **kwargs,
        )

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
