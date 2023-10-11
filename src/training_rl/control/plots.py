import control as ct
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, lti, step

from .environment import create_inverted_pendulum_environment

__all__ = [
    "plot_influence_of_K_on_pendulum",
    "plot_small_angle_approximation",
    "plot_second_order_step_response",
    "plot_estimator_response",
    "plot_mass_spring_damper_results",
    "plot_inverted_pendulum_results",
]


def plot_influence_of_K_on_pendulum(K_values: list[float] | None = None) -> None:
    """Plots the influence of feedback value K for different values of K on the invereted pendulum system."""
    if K_values is None:
        K_values = [1.0, 2.0, 10.0, 20.0]
    env = create_inverted_pendulum_environment()
    initial_observation, _ = env.reset()

    all_observations = []

    for K in K_values:
        env.set_state(initial_observation[:2], initial_observation[2:])
        observation = initial_observation.copy()
        observations = [observation]

        for _ in range(300):
            theta = observation[[1]]
            action = K * theta
            observation, _, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            if terminated or truncated:
                env.reset()
                break

        observations = np.stack(observations)
        all_observations.append(observations)

    for i, K in enumerate(K_values):
        plt.plot(
            np.arange(all_observations[i].shape[0]) * env.dt,
            all_observations[i][:, 1],
            label=f"{K=}",
        )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Angle (rad)")


def plot_small_angle_approximation():
    """Plots the small angle approximation
    to the sine and cosine functions over the range [- pi / 4, pi / 4].
    """
    x = np.arange(-np.pi / 4, np.pi / 4, 0.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(x, x, label="$x$")
    ax1.plot(x, np.sin(x), color="r", label=r"$\sin(x)$")
    ax1.legend()
    ax2.hlines(1, x[0], x[-1], label="$1$")
    ax2.plot(x, np.cos(x), color="r", label=r"$\cos(x)$")
    ax2.plot(x, 1 - x**2 / 2, color="orange", label="$1 - \\frac{x^2}{2}$")
    ax2.legend()


def plot_second_order_step_response() -> None:
    """Plots step response of an under-damped second order system.

    This is used to explain parameter identification for a second order system.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    numerator = [1]
    denomenator = [1, 0.5, 1]
    sys = lti(numerator, denomenator)
    t, y = step(sys, X0=0.0, N=100)
    peak_indices, _ = find_peaks(y, prominence=0.1)
    u = np.ones_like(t)
    u[0] = 0.0

    ax1.plot(t, y, label="Y", color="tab:blue")
    ax1.hlines(y[0], t[0], t[-1], label="$Y_0$", alpha=0.5, color="tab:green", linestyle="--")
    ax1.hlines(y[-1], t[0], t[-1], label="$Y_f$", alpha=0.5, color="tab:red", linestyle="--")
    ax1.plot(t[peak_indices], y[peak_indices], "x", color="tab:orange")
    ax1.vlines(
        x=t[peak_indices],
        ymin=y[-1],
        ymax=y[peak_indices],
        label="$A_i$",
        color="tab:orange",
    )

    ax1.annotate(
        text="$T_w$",
        xy=(t[peak_indices[0]], y[peak_indices[0]] + 0.05),
        xycoords="data",
        xytext=(t[peak_indices[1]], y[peak_indices[0]] + 0.05),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", color="tab:red"),
        va="center",
        color="tab:red",
    )
    ax1.set_ylim(-0.1, 1.6)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Response")
    ax1.legend()

    ax2.plot(t, u, label="U", color="tab:blue")
    ax2.hlines(u[0], t[0], t[-1], label="$U_0$", color="tab:green", alpha=0.5, linestyle="--")
    ax2.hlines(u[-1], t[0], t[-1], label="$U_f$", color="tab:red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control")
    ax2.legend()


def plot_estimator_response(
    estimated_response: ct.timeresp.TimeResponseData,
    *,
    labels: list[str],
    observations: NDArray | None = None,
) -> None:
    """As its name suggests, this function plots the response
    of an estimator.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.errorbar(
        estimated_response.time,
        estimated_response.outputs[0],
        estimated_response.states[2],
        fmt="r-",
        label="Estimated",
    )
    if observations is not None:
        ax1.plot(
            estimated_response.time[: len(observations) - 1],
            observations[1:, 0],
            label="Ground Truth",
        )
    ax1.set_xlabel("Time")
    ax1.set_ylabel(labels[0])
    ax1.legend()
    ax2.errorbar(
        estimated_response.time,
        estimated_response.outputs[1],
        estimated_response.states[5],
        fmt="r-",
        label="Estimated",
    )
    if observations is not None:
        ax2.plot(
            estimated_response.time[: len(observations) - 1],
            observations[1:, 1],
            label="Ground Truth",
        )
    ax2.set_xlabel("Time")
    ax2.set_ylabel(labels[1])
    ax2.legend()
    fig.tight_layout()


def plot_mass_spring_damper_results(
    T: NDArray,
    reference: float,
    observations: NDArray,
    actions: NDArray,
) -> None:
    """As its name suggests, this function plots the results
    of a run of the mass spring damper environment.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.plot(T, observations[:, 0])
    ax1.hlines(reference, T[0], T[-1], "r")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    if observations.shape[1] == 2:
        ax2.plot(T, observations[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Velocity")
    ax3.plot(T[1:], actions)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Force")
    fig.tight_layout()


def plot_inverted_pendulum_results(
    T: NDArray,
    reference: float,
    observations: NDArray,
    actions: NDArray,
) -> None:
    """As its name suggests, this function plots the results
    of a run of the inverted pendulum environment.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.plot(T, observations[:, 0])
    ax1.hlines(reference, T[0], T[-1], "r")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Angle")
    if observations.shape[1] == 2:
        ax2.plot(T, observations[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Angular Velocity")
    ax3.plot(T[1:], actions)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Force")
    fig.tight_layout()