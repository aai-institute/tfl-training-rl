from enum import Enum

from training_rl.offline_rl.behavior_policies.custom_2d_grid_policy import (
    behavior_policy_8x8_grid_avoid_vertical_obstacle,
    behavior_policy_8x8_grid_deterministic_4_0_to_7_7,
    behavior_policy_8x8_grid_moves_downwards_within_strip,
    behavior_policy_8x8_grid_moving_towards_lower_right,
    behavior_policy_8x8_grid_random_towards_left_within_strip,
    horizontal_random_walk, move_right, behavior_policy_8x8_grid_epsilon_greedy_4_0_to_7_7)


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class BehaviorPolicyRestorationConfigFactoryRegistry(CallableEnum):
    behavior_8x8_moving_towards_lower_right = behavior_policy_8x8_grid_moving_towards_lower_right
    behavior_8x8_moves_downwards_within_strip = (
        behavior_policy_8x8_grid_moves_downwards_within_strip
    )
    behavior_8x8_deterministic_4_0_to_7_7 = behavior_policy_8x8_grid_deterministic_4_0_to_7_7
    behavior_8x8_eps_greedy_4_0_to_7_7 = behavior_policy_8x8_grid_epsilon_greedy_4_0_to_7_7

    behavior_8x8_random_towards_left_within_strip = (
        behavior_policy_8x8_grid_random_towards_left_within_strip
    )
    behavior_8x8_avoid_vertical_obstacle = behavior_policy_8x8_grid_avoid_vertical_obstacle
    behavior_move_right = move_right
    behavior_horizontal_random_walk = horizontal_random_walk
    random = lambda action_space: action_space.sample()


class BehaviorPolicyType(str, Enum):
    behavior_8x8_moving_towards_lower_right = "behavior_8x8_moving_towards_lower_right"
    behavior_8x8_deterministic_4_0_to_7_7 = "behavior_8x8_deterministic_4_0_to_7_7"
    behavior_8x8_moves_downwards_within_strip = "behavior_8x8_moves_downwards_within_strip"
    behavior_8x8_random_towards_left_within_strip = "behavior_8x8_random_towards_left_within_strip"
    behavior_8x8_avoid_vertical_obstacle = "behavior_8x8_avoid_vertical_obstacle"
    behavior_move_right = "behavior_move_right"
    behavior_horizontal_random_walk = "behavior_horizontal_random_walk"
    random = "random"
    behavior_8x8_eps_greedy_4_0_to_7_7 = "behavior_8x8_eps_greedy_4_0_to_7_7"
