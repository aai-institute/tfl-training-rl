from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import List, Optional

import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.registration import register as gymnasium_register

from training_rl.offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import \
    ObstacleTypes
from training_rl.offline_rl.custom_envs.utils import InitialConfigCustom2DGridEnvWrapper, Grid2DInitialConfig

ENTRY_POINT_2D_GRID = (
    "training_rl.offline_rl.custom_envs.custom_2d_grid_env.simple_grid:Custom2DGridEnv"
)
ENTRY_POINT_TORCS = (
    "training_rl.offline_rl.custom_envs.gym_torcs.gym_torcs:TorcsEnv"
)
ENTRY_POINT_TORCS_LIDAR_WRAPPER = (
    "training_rl.offline_rl.custom_envs.gym_torcs.gym_torcs:TorcsLidarEnv"
)


class RenderMode(StrEnum):
    RGB_ARRAY_LIST = "rgb_array_list"
    RGB_ARRAY = "rgb_array"
    HUMAN = "human"


class CustomEnv(StrEnum):
    HalfCheetah_v5 = "HalfCheetah-v5"
    Grid_2D_4x4_discrete = "Grid_2D_4x4_discrete"
    Grid_2D_4x4_continuous = "Grid_2D_4x4_continuous"
    Grid_2D_6x6_discrete = "Grid_2D_6x6_discrete"
    Grid_2D_6x6_continuous = "Grid_2D_6x6_continuous"
    Grid_2D_8x8_continuous = "Grid_2D_8x8_continuous"
    Grid_2D_8x8_discrete = "Grid_2D_8x8_discrete"
    torcs = "torcs"


@dataclass
class GridEnvConfig:
    env_name: str
    obstacles: Optional[List[str]] = None
    render_mode: RenderMode = None
    discrete_action: bool = True
    max_episode_steps: int = 60

    def __post_init__(self):
        if self.obstacles:
            # Compute GRID_DIMS from OBSTACLES if obstacles are provided
            num_rows = len(self.obstacles)
            num_cols = len(self.obstacles[0]) if num_rows > 0 else 0
            if not (num_rows > 0 and num_cols > 0):
                raise ValueError("To use obstacle maps The grid must be two dimensional!")


def register_custom_grid_env(grid_env_config: GridEnvConfig):
    gymnasium_register(
        id=grid_env_config.env_name,
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=grid_env_config.max_episode_steps,
        kwargs={
            "obstacle_map": grid_env_config.obstacles,
            "discrete_action": grid_env_config.discrete_action,
        },
    )


def register_torcs_env(vision=False, throttle=False, gear_change=False):
    gymnasium_register(
        id="torcs_original",
        entry_point=ENTRY_POINT_TORCS,
        kwargs={
            "vision": vision,
            "throttle": throttle,
            "gear_change": gear_change,
        }
    )


# ToDo: I could use a wrapper but minari complains as it checks the unwrapped env .
def register_torcs_lidar_version_env(vision=False, throttle=False, gear_change=False):
    gymnasium_register(
        id="torcs",
        entry_point=ENTRY_POINT_TORCS_LIDAR_WRAPPER,
        kwargs={
            "vision": vision,
            "throttle": throttle,
            "gear_change": gear_change,
        }
    )


def register_HalfCheetah_v5_env(max_episode_steps=50):
    env_name = CustomEnv.HalfCheetah_v5
    entry_point = "gymnasium.envs.mujoco:HalfCheetahEnv"
    reward_threshold = 4800.0

    return gymnasium_register(
        id=env_name,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        reward_threshold=reward_threshold,
    )


def register_custom_grid_envs(
    env_name: CustomEnv,
    obstacles: ObstacleTypes,
    discrete_action: bool,
    render_mode: RenderMode,
):
    config = {
        "env_name": env_name,
        "obstacles": obstacles,
        "discrete_action": discrete_action,
        "render_mode": render_mode,
    }
    config = GridEnvConfig(**config)
    register_custom_grid_env(config)


def register_grid_envs():
    max_episode_steps = 5
    register_HalfCheetah_v5_env(max_episode_steps=max_episode_steps)

    register_torcs_env()

    register_torcs_lidar_version_env()

    obstacles = ObstacleTypes.obst_free_4x4.value
    register_custom_grid_envs(
        CustomEnv.Grid_2D_4x4_discrete, obstacles, True, RenderMode.RGB_ARRAY_LIST
    )
    register_custom_grid_envs(
        CustomEnv.Grid_2D_4x4_continuous, obstacles, False, RenderMode.RGB_ARRAY_LIST
    )

    obstacles = ObstacleTypes.obst_free_6x6.value
    register_custom_grid_envs(
        CustomEnv.Grid_2D_6x6_discrete, obstacles, True, RenderMode.RGB_ARRAY_LIST
    )
    register_custom_grid_envs(
        CustomEnv.Grid_2D_6x6_continuous, obstacles, False, RenderMode.RGB_ARRAY_LIST
    )

    obstacles = ObstacleTypes.obst_free_8x8.value
    register_custom_grid_envs(
        CustomEnv.Grid_2D_8x8_discrete, obstacles, True, RenderMode.RGB_ARRAY_LIST
    )
    register_custom_grid_envs(
        CustomEnv.Grid_2D_8x8_continuous, obstacles, False, RenderMode.RGB_ARRAY_LIST
    )


class EnvFactory(StrEnum):
    HalfCheetah_v5 = CustomEnv.HalfCheetah_v5
    Grid_2D_4x4_discrete = CustomEnv.Grid_2D_4x4_discrete
    Grid_2D_4x4_continuous = CustomEnv.Grid_2D_4x4_continuous
    Grid_2D_6x6_discrete = CustomEnv.Grid_2D_6x6_discrete
    Grid_2D_6x6_continuous = CustomEnv.Grid_2D_6x6_continuous
    Grid_2D_8x8_continuous = CustomEnv.Grid_2D_8x8_continuous
    Grid_2D_8x8_discrete = CustomEnv.Grid_2D_8x8_discrete
    torcs = CustomEnv.torcs

    def get_env(
            self,
            render_mode: RenderMode | None = None,
            grid_config: Grid2DInitialConfig = None
    ) -> Env:
        register_grid_envs()
        match self:
            case self.Grid_2D_4x4_discrete | self.Grid_2D_4x4_continuous | self.Grid_2D_6x6_discrete | \
                    self.Grid_2D_6x6_continuous | self.Grid_2D_8x8_discrete | self.Grid_2D_8x8_continuous:
                return InitialConfigCustom2DGridEnvWrapper(
                    gym.make(self, render_mode=render_mode),
                    env_config=grid_config
                )
            # ToDo: Render mode should be passed here?
            case self.HalfCheetah_v5 | self.torcs:
                return gym.make(self)
