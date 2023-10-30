import logging
import sys
from typing import Callable, Union

import gymnasium as gym
import numpy as np
import pygame
import torch
from matplotlib import pyplot as plt

from tianshou.data import Batch
from tianshou.policy import BasePolicy, ImitationPolicy
from training_rl.offline_rl.behavior_policies.behavior_policy_registry import (
    BehaviorPolicyRestorationConfigFactoryRegistry, BehaviorPolicyType)
from training_rl.offline_rl.custom_envs.custom_envs_registration import \
    RenderMode
from training_rl.offline_rl.custom_envs.utils import (
    Grid2DInitialConfig, InitialConfigCustom2DGridEnvWrapper)
from training_rl.offline_rl.utils import extract_dimension

logging.basicConfig(level=logging.WARNING)


def snapshot_env(env: gym.Env):
    env.reset()
    env.step(0)
    rendered_data = env.render()
    rendered_data = rendered_data[0].reshape(256, 256, 3)
    plt.imshow(rendered_data)
    plt.show()


def render_rgb_frames_pygame(env: gym.Env, screen, time_frame=20):
    clock = pygame.time.Clock()
    desired_fps = time_frame
    clock.tick(desired_fps)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    rendered_data = env.render()
    frames = np.transpose(rendered_data[0], (1, 0, 2))
    pygame_surface = pygame.surfarray.make_surface(frames)
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()


def initialize_pygame(title="RL agent animation"):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode([256, 256])
    pygame.display.set_caption(title)
    return screen


def offpolicy_rendering(
    env_or_env_name: Union[gym.Env, str],
    render_mode: RenderMode,
    env_2d_grid_initial_config: Grid2DInitialConfig = None,
    behavior_policy_name: BehaviorPolicyType = None,
    policy_model: Union[BasePolicy, Callable] = None,
    num_frames: int = 100,
    imitation_policy_sampling: bool = False,
):
    """
    :param env_or_env_name: A gym environment or an env name.
    :param render_mode:
    :param env_2d_grid_initial_config: Initial config, namely obstacles and initial and target positions. Only used
        for Custom2DGridEnv configuration when env_or_env_name is the registered environment name.
    :param behavior_policy_name: name of behavior policy (only if behavior_policy is None -
        see behavior_policy_registry.py)
    :param policy_model: A Tianshou policy mode or a callable that accept an state and the env and returns an action
    :param num_frames: Number of frames
    :param imitation_policy_sampling: Only for imitation learning policy. If False we compute the eps greedy of \pi(a|s).
    :return:

    Usage:
    ```
    register_grid_envs()

    env_2D_grid_initial_config = Grid2DInitialConfig(
        obstacles=ObstacleTypes.obst_middle_8x8,
        initial_state=(0,0),
        target_state=(7,7),
    )

    behavior_policy_rendering(
        env_name=CustomEnv.Grid_2D_8x8_discrete,
        render_mode=RenderMode.RGB_ARRAY_LIST,
        behavior_policy_name=BehaviorPolicyType.behavior_suboptimal_8x8_grid_discrete,
        env_2d_grid_initial_config=env_2D_grid_initial_config,
        num_frames=1000,
    )
    ```

    """
    if behavior_policy_name is None and policy_model is None:
        raise ValueError("Either behavior_policy_name or behavior_policy must be provided.")
    if behavior_policy_name is not None and policy_model is not None:
        raise ValueError(
            "Both behavior_policy_name and behavior_policy cannot be provided simultaneously."
        )

    if isinstance(env_or_env_name, str):
        env = InitialConfigCustom2DGridEnvWrapper(
            gym.make(env_or_env_name, render_mode=render_mode),
            env_config=env_2d_grid_initial_config,
        )
    else:
        env = env_or_env_name

    state, _ = env.reset()

    state_shape = extract_dimension(env.observation_space)

    if render_mode == RenderMode.RGB_ARRAY_LIST:
        screen = initialize_pygame()

    for _ in range(num_frames):
        if behavior_policy_name is not None:
            behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[
                behavior_policy_name
            ]

            if behavior_policy_name == BehaviorPolicyType.random:
                action = env.action_space.sample()
            else:
                action = behavior_policy(state, env)
        else:
            if isinstance(policy_model, BasePolicy):
                tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
                policy_output = policy_model(tensor_state)

                if imitation_policy_sampling and isinstance(policy_model, ImitationPolicy):
                    policy_output = policy_output.logits
                    categorical = torch.distributions.Categorical(logits=policy_output[0])
                    action = np.array(categorical.sample())
                else:
                    action = (
                        policy_output.act[0]
                        if (
                            isinstance(policy_output.act[0], np.ndarray)
                            or isinstance(policy_output.act, np.ndarray)
                        )
                        else policy_output.act[0].detach().numpy()
                    )
            elif isinstance(policy_model, Callable):
                action = policy_model(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_frames += 1

        if render_mode == RenderMode.RGB_ARRAY_LIST:
            render_rgb_frames_pygame(env, screen)
        else:
            env.render()

        if done or time_out:
            state, _ = env.reset()
            num_frames = 0
        else:
            state = next_state

    pygame.quit()
