from enum import Enum

from training_rl.offline_rl.behavior_policies.custom_2d_grid_policy import (
    behavior_policy_8x8_grid_avoid_vertical_obstacle,
    behavior_policy_8x8_grid_deterministic_0_0_to_4_7,
    behavior_policy_8x8_grid_deterministic_4_0_to_7_7,
    behavior_policy_8x8_grid_epsilon_greedy_4_0_to_7_7,
    behavior_policy_8x8_grid_moves_downwards_within_strip,
    behavior_policy_8x8_grid_moves_downwards_within_strip_and_left,
    behavior_policy_8x8_grid_moving_towards_lower_right,
    behavior_policy_8x8_grid_random_towards_left_within_strip,
    behavior_policy_8x8_grid_suboptimal_0_0_to_4_7, horizontal_random_walk,
    move_right, behavior_policy_8x8_suboptimal_determ_initial_3_0_final_3_7,
    behavior_policy_8x8_suboptimal_rnd_initial_3_0_final_3_7, behavior_policy_8x8_suboptimal_initial_0_0_final_0_7,
    move_up_from_bottom_5_steps, move_from_7_7_twice_to_left, move_up, move_left_with_noise)
from training_rl.offline_rl.behavior_policies.custom_torcs_policy import get_torcs_expert_policy, \
    get_torcs_drunk_driver_policy, get_torcs_expert_policy_with_noise


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

# ToDo: This factory should be refactor soon!


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
    behavior_move_up_from_bottom_5_steps = move_up_from_bottom_5_steps
    behavior_horizontal_random_walk = horizontal_random_walk
    random = lambda action_space: action_space.sample()
    behavior_8x8_moves_downwards_within_strip_and_left = (
        behavior_policy_8x8_grid_moves_downwards_within_strip_and_left
    )
    behavior_8x8_grid_deterministic_0_0_to_4_7 = behavior_policy_8x8_grid_deterministic_0_0_to_4_7
    behavior_8x8_grid_suboptimal_0_0_to_4_7 = behavior_policy_8x8_grid_suboptimal_0_0_to_4_7
    torcs_expert_policy = get_torcs_expert_policy
    torcs_expert_policy_with_noise = get_torcs_expert_policy_with_noise
    torcs_drunk_driver_policy = get_torcs_drunk_driver_policy
    behavior_8x8_suboptimal_determ_initial_3_0_final_3_7 = behavior_policy_8x8_suboptimal_determ_initial_3_0_final_3_7
    behavior_8x8_suboptimal_rnd_initial_3_0_final_3_7 = behavior_policy_8x8_suboptimal_rnd_initial_3_0_final_3_7
    behavior_8x8_suboptimal_initial_0_0_final_0_7 = behavior_policy_8x8_suboptimal_initial_0_0_final_0_7
    behavior_8x8_move_from_7_7_twice_to_left = move_from_7_7_twice_to_left
    behavior_8x8_move_up = move_up
    behavior_8x8_move_left_with_noise = move_left_with_noise


class BehaviorPolicyType(str, Enum):
    behavior_8x8_moving_towards_lower_right = "behavior_8x8_moving_towards_lower_right"
    behavior_8x8_deterministic_4_0_to_7_7 = "behavior_8x8_deterministic_4_0_to_7_7"
    behavior_8x8_eps_greedy_4_0_to_7_7 = "behavior_8x8_eps_greedy_4_0_to_7_7"
    behavior_8x8_moves_downwards_within_strip = "behavior_8x8_moves_downwards_within_strip"
    behavior_8x8_random_towards_left_within_strip = "behavior_8x8_random_towards_left_within_strip"
    behavior_8x8_avoid_vertical_obstacle = "behavior_8x8_avoid_vertical_obstacle"
    behavior_move_right = "behavior_move_right"
    behavior_move_up_from_bottom_5_steps = "behavior_move_up_from_bottom_5_steps"
    behavior_horizontal_random_walk = "behavior_horizontal_random_walk"
    random = "random"
    behavior_8x8_moves_downwards_within_strip_and_left = (
        "behavior_8x8_moves_downwards_within_strip_and_left"
    )
    behavior_8x8_grid_deterministic_0_0_to_4_7 = "behavior_8x8_grid_deterministic_0_0_to_4_7"
    behavior_8x8_grid_suboptimal_0_0_to_4_7 = "behavior_8x8_grid_suboptimal_0_0_to_4_7"
    torcs_expert_policy = "torcs_expert_policy"
    torcs_drunk_driver_policy = "torcs_drunk_driver_policy"
    torcs_expert_policy_with_noise = "torcs_expert_policy_with_noise"
    behavior_8x8_suboptimal_determ_initial_3_0_final_3_7 = "behavior_8x8_suboptimal_determ_initial_3_0_final_3_7"
    behavior_8x8_suboptimal_rnd_initial_3_0_final_3_7 = "behavior_8x8_suboptimal_rnd_initial_3_0_final_3_7"
    behavior_8x8_suboptimal_initial_0_0_final_0_7 = "behavior_8x8_suboptimal_initial_0_0_final_0_7"
    behavior_8x8_move_from_7_7_twice_to_left = "behavior_8x8_move_from_7_7_twice_to_left"
    behavior_8x8_move_up = "behavior_8x8_move_up"
    behavior_8x8_move_left_with_noise = "behavior_8x8_move_left_with_noise"
