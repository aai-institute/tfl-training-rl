from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
from gymnasium import Env, spaces

import training_rl.offline_rl.custom_envs.custom_2d_grid_env.rendering as r
from training_rl.offline_rl.custom_envs.custom_2d_grid_env.window import Window


def integer_to_one_hot(integer_value, n=64):
    if integer_value < 0 or integer_value > n:
        raise ValueError("Integer value is out of range [0, n]")

    one_hot_vector = np.zeros(n + 1)
    one_hot_vector[integer_value] = 1
    return one_hot_vector


def one_hot_to_integer(one_hot_vector):
    if not isinstance(one_hot_vector, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if len(one_hot_vector.shape) != 1:
        raise ValueError("Input must be a 1-dimensional array")

    return np.argmax(one_hot_vector)


class Custom2DGridEnv(Env):
    """
    Simple Grid Environment: Adapted from https://github.com/damat-le/gym-simplegrid.git for use in
    the RL workshop, with custom modifications to meet specific workshop requirements.

    The environment is a grid with obstacles (walls). The agent can move in one of the four cardinal directions.
    If they try to move over an obstacle or out of the grid bounds, they stay in place. The environment is
    episodic, i.e. the episode ends when the agent reaches its goal.

    To initialise the grid, the user needs to send a mask with 0's (obstacle free) and 1's(obstacles), e.g.

    ["0000",
     "0101",
     "0001",
     "1000"]

    actions: They are by default discrete, i.e. gym.spaces.Discete(4) but we will make them also continuous
    like spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64) just to use the same environment
    with continuous action RL algorithms. However, the four dimensional vector will be converted to a
    one-hot encoded one in order to mimic a discrete environment (see explanation in __init__ below)

    observations: They should be integers representing the agent's position in the gird. However, we will
    use again a vector (one-hot encoded) in order to use the environment with RL algorithms already implemented
    in Tianshou that deal with vectorial observation spaces.

    The user can also decide the starting and goal positions of the agent. This can be done by through
    the `options` dictionary in the `reset` method. The user can specify the starting and goal positions
    by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type
    int (s) or tuple (x,y) and represent the agent starting and goal positions respectively.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
    MOVES: dict[int, tuple] = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),  # RIGHT
    }

    def __init__(
        self,
        obstacle_map: list[str],
        render_mode: str | None = None,
        discrete_action: bool = True,
        vectorial_observation_space: bool = True,
    ):
        """
        :param obstacle_map:
        :param render_mode:
        :param discrete_action: If False, this is a trick to make the environment work with continuous
            action spaces RL algorithms. In the discrete case the action will be 0,1,2 or 3 but in the
            continuous we will have a vector, i.e. action = np.array([a_1,a_2,a_2,a_3]). However we will
            always convert it to one-hot encoding in order to mimic a discrete action space.
            E.g. action = np.array([0.5,0.1,0.2,0.6]) -> action = np.array([0.0,0.0,0.0,1.0])
         :param vectorial_observation_space: if True a one-hot encoding format will be used as this will be
            useful for RL algorithms that deal with vectorial observations.
        """

        # Env confinguration
        self.obstacles = self.parse_obstacle_map(obstacle_map)  # walls
        self.nrow, self.ncol = self.obstacles.shape

        self.discrete_action = discrete_action
        if self.discrete_action:
            self.action_space = spaces.Discrete(
                len(self.MOVES)
            )  # Two possible actions: move left or move right
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)

        self.vectorial_observation_space = vectorial_observation_space
        if self.vectorial_observation_space:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.nrow * self.ncol,), dtype=np.float64
            )
        else:
            self.observation_space = spaces.Discrete(self.nrow * self.ncol - 1)

            # Rendering configuration
        self.render_mode = render_mode
        self.window = None
        self.agent_color = "yellow"
        self.tile_cache = {}
        self.fps = self.metadata["render_fps"]
        self.num_steps = 0

        start_location = (0, 0)
        target_location = (self.ncol - 1, self.nrow - 1)
        options = {"start_loc": start_location, "goal_loc": target_location}
        self.start_xy = self.parse_state_option("start_loc", options)
        self.goal_xy = self.parse_state_option("goal_loc", options)

    def reset(self, seed: int | None = None, options: dict = None) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()

        # Check integrity
        self.integrity_checks()

        # if self.render_mode == "human":
        self.render()

        self.num_steps = 0

        return self.get_obs(), self.get_info()

    def step(self, action: Union[int, np.ndarray]):
        """
        Take a step in the environment.
        """

        if isinstance(action, np.ndarray):
            if len(action.shape) > 0:
                action = int(action.argmax())  # if action is a vector of directions
            else:
                action = int(action)  # if action is an scalar with the

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)

        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        self.render()
        self.num_steps += 1

        time_out = False

        if self.get_obs()[1] == 4:
            print(self.get_obs(), action)

        return self.get_obs(), self.reward, self.done, time_out, self.get_info()

    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        #>>> my_map = ["001", "010", "011]
        #>>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype="c")
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        # elif isinstance(obstacle_map, str):
        #    map_str = MAPS[obstacle_map]
        #    map_str = np.asarray(map_str, dtype='c')
        #    map_int = np.asarray(map_str, dtype=int)
        #    return map_int
        # else:
        #    raise ValueError(f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}.")

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple) or isinstance(state, list):
                return state
            else:
                raise TypeError(f"Allowed types for `{state_name}` are int or tuple.")
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(
                f"Key `{state_name}` not found in `options`. Random sampling a valid value for it:"
            )
            logger.info(f"...`{state_name}` has value: {state}")
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.observation_space.sample()
        state = one_hot_to_integer(state)
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.observation_space.sample()
            state = one_hot_to_integer(state)
            pos_xy = self.to_xy(state)
        return pos_xy

    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert (
            self.obstacles[tuple(self.start_xy)] == 0
        ), f"Start position {self.start_xy} overlaps with a wall."
        assert (
            self.obstacles[tuple(self.goal_xy)] == 0
        ), f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(
            *tuple(self.start_xy)
        ), f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(
            *tuple(self.goal_xy)
        ), f"Goal position {self.goal_xy} is out of bounds."

    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return tuple(self.agent_xy) == tuple(self.goal_xy)

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == 0

    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        # if not self.is_in_bounds(x, y):
        #    return -1.0
        # elif not self.is_free(x, y):
        #    return -1.0
        # if (x, y) == self.goal_xy:
        #    return 0.0

        # rew = -1.0
        # if y != 0:
        #    rew += -1.0
        # else:
        #    rew += -0.1

        # return rew
        if (x, y) == self.goal_xy:
            return 0.0
        elif not self.is_in_bounds(x, y):
            return -1.0
        else:
            return (
                -0.1
                * ((self.goal_xy[0] - x) ** 2 + (self.goal_xy[1] - y) ** 2) ** 0.5
                / (self.ncol * self.nrow)
            )

    def get_obs(self) -> int:
        state = self.to_s(*self.agent_xy)
        state = integer_to_one_hot(state, n=self.ncol * self.nrow - 1)
        return state

    def get_info(self) -> dict:
        return {"agent_xy": self.agent_xy}

    def close(self):
        """
        Close the environment.
        """
        if self.window:
            self.window.close()
        return None

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            pass
        elif self.render_mode == "human":
            img = self.render_frame()
            if not self.window:
                self.window = Window()
                self.window.show(block=False)
            caption = ""
            self.window.show_img(img, caption, self.fps)
            return None
        elif self.render_mode == "rgb_array":
            return self.render_frame()
        elif self.render_mode == "rgb_array_list":
            img = self.render_frame()
            self.frames.append(img)
            return self.frames
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self, tile_size=r.TILE_PIXELS, highlight_mask=None):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly
        render it, we have to pass (y,x) to the grid.render method.

        tile_size: tile size in pixels
        """
        width = self.ncol
        height = self.nrow

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render grid with obstacles
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.obstacles[x, y] == 1:
                    cell = r.Wall(color="black")
                else:
                    cell = None

                img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render start
        x, y = self.start_xy
        cell = r.ColoredTile(color="red")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render goal
        x, y = self.goal_xy
        cell = r.ColoredTile(color="green")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render agent
        x, y = self.agent_xy
        cell = r.Agent(color=self.agent_color)
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        return img

    def render_cell(self, obj: r.WorldObj, highlight=False, tile_size=r.TILE_PIXELS, subdivs=3):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        if not isinstance(obj, r.Agent):
            key = (None, highlight, tile_size)
            key = obj.encode() + key if obj else key

            if key in self.tile_cache:
                return self.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) + 255

        if obj != None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            r.highlight_img(img)

        # Draw the grid lines (top and left edges)
        r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (170, 170, 170))
        r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (170, 170, 170))

        # Downsample the image to perform supersampling/anti-aliasing
        img = r.downsample(img, subdivs)

        # Cache the rendered tile
        if not isinstance(obj, r.Agent):
            self.tile_cache[key] = img

        return img

    def update_cell_in_frame(self, img, x, y, cell, tile_size):
        """
        Parameters
        ----------
        img : np.ndarray
            Image to update.
        x : int
            x-coordinate of the cell to update.
        y : int
            y-coordinate of the cell to update.
        cell : r.WorldObj
            New cell to render.
        tile_size : int
            Size of the cell in pixels.
        """
        tile_img = self.render_cell(cell, tile_size=tile_size)
        height_min = x * tile_size
        height_max = (x + 1) * tile_size
        width_min = y * tile_size
        width_max = (y + 1) * tile_size
        img[height_min:height_max, width_min:width_max, :] = tile_img
        return img

    def set_new_obstacle_map(self, new_obstacle_map):
        self.obstacles = self.parse_obstacle_map(new_obstacle_map)

    def set_starting_point(self, start_xy: Tuple[int, int]):
        options = {"start_loc": start_xy, "goal_loc": self.goal_xy}
        self.start_xy = self.parse_state_option("start_loc", options)

    def set_goal_point(self, goal_xy: Tuple[int, int]):
        options = {"start_loc": self.start_xy, "goal_loc": goal_xy}
        self.goal_xy = self.parse_state_option("goal_loc", options)
