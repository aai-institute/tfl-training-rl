import random

import numpy as np

from training_rl.offline_rl.custom_envs.custom_2d_grid_env.simple_grid import \
    Custom2DGridEnv
from training_rl.offline_rl.utils import one_hot_to_integer

# MOVES:
#   0: (-1, 0),  # UP
#   1: (1, 0),  # DOWN
#   2: (0, -1),  # LEFT
#   3: (0, 1)  # RIGHT


def behavior_policy_8x8_grid_moving_towards_lower_right(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    This policy makes the agent move to the lower-right corner of the grid.

    :param state: Agent state
    :param env:
    :return: The action

    """

    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    possible_directions = [2, 3, 1]
    weights = [1, 1, 3]
    random_directions = random.choices(possible_directions, weights=weights)[0]

    if state_xy[0] == 7:
        possible_directions = [0, 1, 3]
        weights = [1, 1, 3]
        random_directions = random.choices(possible_directions, weights=weights)[0]

    return random_directions


def behavior_policy_8x8_grid_moves_downwards_within_strip(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    This policy moves the agent downward and left-right but with limited horizontal mobility,
    constrained to the first three cells of the grid

    :param state: Agent state
    :param env:
    :return: The action
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [2, 3, 1]
    weights = [1, 1, 1]
    random_directions = random.choices(possible_directions, weights=weights)[0]
    if random_directions == 3 and (state_xy[1] > 2):
        possible_directions = [2, 1]
        weights = [1, 1]
        return random.choices(possible_directions, weights=weights)[0]

    return random_directions


def behavior_policy_8x8_grid_deterministic_4_0_to_7_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    Deterministic suboptimal policy to move agent from (4,0) towards (7,7)

    :param state: Agent state
    :param env:
    :return: The action
    :rtype:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[0] == 4 and state_xy[1] < 7:
        return 3
    else:
        return 1


def behavior_policy_8x8_grid_deterministic_0_0_to_4_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    Deterministic suboptimal policy to move agent from (4,0) towards (7,7)

    :param state: Agent state
    :param env:
    :return: The action
    :rtype:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[1] == 7:
        return 0
    elif state_xy[0] == 4:
        return 3
    else:
        return 1


def behavior_policy_8x8_grid_suboptimal_0_0_to_4_7(state: np.ndarray, env: Custom2DGridEnv) -> int:
    """
    Deterministic suboptimal policy to move agent from (4,0) towards (7,7)

    :param state: Agent state
    :param env:
    :return: The action
    :rtype:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    if state_xy == (5, 7):
        return 0

    if state_xy[0] == 4:
        return 3

    possible_directions = [1, 2, 3]
    weights = [1, 1, 1]
    random_directions = random.choices(possible_directions, weights=weights)[0]
    return random_directions


def behavior_policy_8x8_grid_epsilon_greedy_4_0_to_7_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    Deterministic suboptimal policy to move agent from (4,0) towards (7,7)

    :param state: Agent state
    :param env:
    :return: The action
    :rtype:
    """

    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy == (4, 0):
        action = 1
    elif state_xy[0] == 5 and state_xy[1] < 7:
        action = 3
    else:
        action = 1

    epsilon = 0.5
    if random.random() < epsilon:
        possible_directions = [0, 1, 3]
        weights = [3, 3, 1]
        action = random.choices(possible_directions, weights=weights)[0]

    return action


def behavior_policy_8x8_grid_random_towards_left_within_strip(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    Moves agent stochastically towards left on the upper half of the grid
    :param state:
    :param env:
    :return:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [0, 1, 2, 3]
    weights = [1, 1, 1, 2]

    if state_xy[0] == 4:
        weights = [2, 1, 2]
        possible_directions = [0, 2, 3]

    random_directions = random.choices(possible_directions, weights=weights)[0]

    return random_directions


def behavior_policy_8x8_grid_avoid_vertical_obstacle(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    moves the agent around ObstacleTypes.verical_object_8x8

    :param state:
    :param env:
    :return:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    possible_directions = [0, 1, 2, 3]
    weights = [1, 1, 1, 1]

    if state_xy[0] < 4 and state_xy[1] < 4:
        weights = [0, 1, 0, 0]
    if state_xy[0] >= 4:
        weights = [1, 1, 1, 2]
    if state_xy[0] >= 4 and state_xy[1] >= 4:
        weights = [2, 1, 1, 2]

    random_directions = random.choices(possible_directions, weights=weights)[0]

    return random_directions


def move_right(state: np.ndarray, env: Custom2DGridEnv) -> int:
    return 3


def horizontal_random_walk(state: np.ndarray, env: Custom2DGridEnv) -> int:
    possible_directions = [2, 3]
    weights = [1, 1]
    return random.choices(possible_directions, weights=weights)[0]


def behavior_policy_8x8_grid_moves_downwards_within_strip_and_left(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    """
    This policy moves the agent downward and left-right but with limited horizontal mobility,
    constrained to the first three cells of the grid

    :param state: Agent state
    :param env:
    :return: The action
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [2, 3, 1]
    weights = [1, 1, 1]
    random_directions = random.choices(possible_directions, weights=weights)[0]

    if state_xy[0] == 7:
        return 3

    if random_directions == 3 and (state_xy[1] > 2):
        possible_directions = [2, 1]
        weights = [1, 1]
        return random.choices(possible_directions, weights=weights)[0]

    return random_directions


def behavior_policy_8x8_suboptimal_determ_initial_3_0_final_3_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[1] == 7:
        action = 0
    elif state_xy[0] == 7:
        action = 3
    elif 3 <= state_xy[0] <= 7:
        action = 1

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]
    else:
        return action


def behavior_policy_8x8_suboptimal_rnd_initial_3_0_final_3_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [0, 1, 2, 3]
    weights = [0.6, 0.6, 0.6, 1.0]
    action = random.choices(possible_directions, weights=weights)[0]
    if state_xy[0] <= 3:
        action = 1

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]
    return action


def behavior_policy_8x8_suboptimal_initial_0_0_final_0_7(
    state: np.ndarray, env: Custom2DGridEnv
) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    if state_xy[1] == 6:
        return 0
    if state_xy[0] < 4:
        return 1
    else:
        return 3


def move_from_7_7_twice_to_left(state: np.ndarray, env: Custom2DGridEnv) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[1] < 5:
        return 3
    else:
        return 2


def move_up(state: np.ndarray, env: Custom2DGridEnv) -> int:
    action = 0
    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]
    else:
        return action


def move_left_with_noise(state: np.ndarray, env: Custom2DGridEnv) -> int:
    possible_directions = [0, 1, 2, 3]
    weights = [1, 1, 4, 1]

    action = random.choices(possible_directions, weights=weights)[0]

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]
    else:
        return action


def move_up_from_bottom_5_steps(state: np.ndarray, env: Custom2DGridEnv) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[0] > 2:
        return 0
    else:
        return 1


# MOVES:
#   0: (-1, 0),  # UP
#   1: (1, 0),  # DOWN
#   2: (0, -1),  # LEFT
#   3: (0, 1)  # RIGHT

