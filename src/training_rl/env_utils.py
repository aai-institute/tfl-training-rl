from dataclasses import dataclass
from typing import Callable, Literal

import gymnasium
import gymnasium as gym
import numpy as np
from celluloid import Camera
from IPython.core.display import HTML
from IPython.core.display_functions import display
from matplotlib import pyplot as plt


@dataclass
class TrajEntry:
    step: int
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray

    frame: np.ndarray | None = None


TTrajectory = list[TrajEntry]


def collect_trajectory(
    env: gym.Env, agent: Callable[[np.ndarray], np.ndarray] | Literal["random"] = "random"
):
    """Agent that samples random actions from the environment."""
    obs, info = env.reset()
    trajectory = []
    step = 0
    while True:
        if agent == "random":
            action = env.action_space.sample()
        else:
            action = agent(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        trajectory.append(
            TrajEntry(
                step=step,
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                frame=frame,
            )
        )
        obs = next_obs
        step += 1
        if done or truncated:
            break
    return trajectory


def compute_monte_carlo_return(trajectory: TTrajectory, gamma: float = 0.99, t=0):
    """Compute the discounted return from a trajectory."""
    trajectory = trajectory[t:]
    return sum([gamma**i * entry.reward for i, entry in enumerate(trajectory)])


def get_trajectory_animation(
    traj: TTrajectory,
    entry_plotter: Callable[[TrajEntry], None],
    title_extractor: Callable[[TrajEntry], str] | str = "Trajectory",
    dpi=150,
    figsize=(3, 3),
    display_frame_count=True,
):
    fig = plt.figure(dpi=dpi, figsize=figsize)
    camera = Camera(fig)
    for id_frame, entry in enumerate(traj):
        title = title_extractor if isinstance(title_extractor, str) else title_extractor(entry)
        entry_plotter(entry)
        ax = plt.gca()
        ax.text(-0.05, 1.05, title, transform=ax.transAxes)
        if display_frame_count:
            ax.text(
                0.75,
                1.05,
                f"frame: {id_frame}",
                fontsize=8,
                bbox=dict(facecolor="gray", fill=True, linewidth=1, boxstyle="round"),
                transform=ax.transAxes,
            )
        camera.snap()
    animation = camera.animate().to_html5_video()
    plt.close()
    display(HTML(animation))


def demo_model(
    env: gymnasium.Env,
    agent: Callable[[np.ndarray], np.ndarray] | Literal["random"],
    num_steps: int = 1000,
):
    concatenated_trajectory = []
    while len(concatenated_trajectory) < num_steps:
        concatenated_trajectory.extend(collect_trajectory(env, agent))
    get_trajectory_animation(concatenated_trajectory, lambda entry: plt.imshow(entry.frame))
