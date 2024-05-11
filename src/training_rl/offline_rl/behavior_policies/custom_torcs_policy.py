import numpy as np


def get_torcs_expert_policy(state: "state", env: "TorcsLidarEnv"=None, noise=False) -> np.ndarray:
    """
    suboptimal expert policy: It is not really an expert policy as it has a hard time to do a good job when the car
        is out of the road.
    """
    steer = state["angle"] * 10 / np.pi
    steer -= state["trackPos"] * 0.10

    #if noise:
    #    noise = np.random.normal(loc=-0.2, scale=2.0,)
    #    steer = np.clip(steer + noise, -1.0, 1.0)

    if noise:
        zigzag_noise = 0.0
        if env is not None:
            time_elapsed = env.number_steps()
            zigzag_noise = np.sin(0.5*time_elapsed)
        noise = np.random.normal(scale=5.0)
        steer = np.clip(steer + zigzag_noise + noise, -0.1, 0.1)

    return np.array([steer])


def get_torcs_expert_policy_with_noise(state: "state", env: "TorcsLidarEnv"=None) -> np.ndarray:
    return get_torcs_expert_policy(state, env, noise=True)


def get_torcs_drunk_driver_policy(state: "state", env: "TorcsEnv"=None) -> np.ndarray:
    #steer = (-1)**(env.number_steps()//1000)*0.05
    if state["trackPos"] < -0.1:
        steer = 0.2
    elif state["trackPos"] > 0.0 and state["trackPos"] <0.1:
        steer = -0.03
    elif state["trackPos"] > 0.1:
        steer = -0.5
    else:
        steer = 0.0
    return np.array([steer])
