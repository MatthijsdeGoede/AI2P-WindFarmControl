from typing import Optional
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gymnasium import spaces
from gymnasium.utils import env_checker

import numpy as np

from GCH_block import GCH

from utils.extract_windspeed import WindspeedExtractor
from utils.preprocessing import read_turbine_positions, angle_to_vec, create_turbine_graph_tensors, correct_angles
from utils.visualization import plot_mean_absolute_speed, get_mean_absolute_speed_figure

device = torch.device("cpu")

class TurbineEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "matplotlib"], "render_fps": 4}

    def __init__(self, windspeedmap_model, turbine_locations, render_mode=None, map_size=128, yaw_step=5, max_yaw=30):
        self.turbine_locations = torch.tensor(turbine_locations)
        self.n_turbines = len(turbine_locations)
        self.map_size = map_size

        # Model to predict the wind speed map
        self.model = windspeedmap_model

        # Extract wind speeds from the map
        self.windspeed_extractor = WindspeedExtractor(turbine_locations, map_size)

        self._n_yaw_steps = (max_yaw // yaw_step) + 1

        # Define some variables to keep track off
        self._wind_direction = [0]
        self._yaws = np.zeros(self.n_turbines, dtype=int)

        # Observations that are available to the agent, the current global wind direction and the current yaw angles
        self.observation_space = spaces.Dict(
            {
                "wind_direction": gym.spaces.Box(0, 360, dtype=int),
                "yaws": spaces.MultiDiscrete([2 * self._n_yaw_steps - 1] * self.n_turbines)
            }
        )

        # Actions the agents can take, it can select the new yaw angles for the turbines
        self.action_space = spaces.MultiDiscrete([2 * self._n_yaw_steps - 1] * self.n_turbines)

        # Convert action to an actual yaw angle
        self._action_to_yaw = lambda action, wind : (action - self._n_yaw_steps) * yaw_step + wind
        self._yaw_to_action = lambda yaw, wind    : (yaw - wind) // yaw_step + self._n_yaw_steps

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_obs(self):
        return {
            "wind_direction": self._wind_direction,
            "yaws": self._yaws,
        }

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is None:
            options = {}

        # If wind direction is given in options, use that
        if "wind_direction" in options:
            self._wind_direction = correct_angles(options["wind_direction"])
        else:
            self._wind_direction =  correct_angles(self.np_random.integers(220, 261, size=(1,), dtype=int))

        if "yaws" in options:
            self._yaws = self._yaw_to_action(correct_angles(options["yaws"]), self._wind_direction[0])
        else:
            # TODO: only use values actually observed in training the wind speed map
            self._yaws = self.np_random.integers(0, 2 * self._n_yaw_steps - 1, size=self.n_turbines)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(action, self._wind_direction[0])

        yaw_angles = list(torch.tensor(yaws).flatten().float()-self._wind_direction[0])

        pos = self.turbine_locations

        x_cor = list(pos[:, 0].flatten())
        y_cor = list(pos[:, 1].flatten())

        power = self.model(x_coordinates_turbines =  x_cor,
                y_coordinates_turbines = y_cor,
                wind_directions = [-1*self._wind_direction[0]+270],
                wind_speeds = [8],
                yaw_angles = yaw_angles,
                return_map = False)

        self._yaws = action

        observation = self._get_obs()
        info = self._get_info()

        # return tuple in form observation, reward, terminated, truncated, info.
        # Power is the reward, and as of now, the environment does not terminate.
        return observation, power, False, False, info

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "matplotlib":
            return self._render_frame()

    def _render_frame(self):
        # Render a wind speed map with current yaws
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(self._yaws, self._wind_direction[0])

        yaw_angles = list(torch.tensor(yaws).flatten().float()-self._wind_direction[0])
        wind_vec = angle_to_vec(self._wind_direction[0])
        pos = self.turbine_locations

        x_cor = list(pos[:, 0].flatten())
        y_cor = list(pos[:, 1].flatten())

        grid,power = self.model(x_coordinates_turbines =  x_cor,
                y_coordinates_turbines = y_cor,
                wind_directions = [-1*self._wind_direction[0]+270],
                wind_speeds = [8],
                yaw_angles = yaw_angles,
                return_map = True)

        wind_speed_map = grid[0][0]

        turbine_pixels = []
        self.windspeed_extractor(wind_speed_map, self._wind_direction, yaws, turbine_pixels)
        wind_vec = 20 * wind_vec


        if self.render_mode == "rgb_array":
            plot_mean_absolute_speed(wind_speed_map, wind_vec, windmill_blades=turbine_pixels)
        if self.render_mode == "matplotlib":
            return get_mean_absolute_speed_figure(wind_speed_map, wind_vec, windmill_blades=turbine_pixels)
        return

def create_env(case=1):
    # Parallel environments
    # Make sure to actually use model that accepts an array of yaw angles instead of this, and load the pretrained weights.
    max_episode_steps = 100

    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../../Data/input_data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    turbine_locations = read_turbine_positions(layout_file)

    model = GCH( x_resolution = 128,
                 y_resolution = 128,
                 x_bounds = (0,5000),
                 y_bounds = (0,5000),
                 x_size = 128,
                 y_size = 128,
                 height = 90)

    env = TimeLimit(TurbineEnv(model, turbine_locations, render_mode="matplotlib", map_size=128), max_episode_steps=max_episode_steps)

    return env


if __name__ == "__main__":
    # Make sure to actually use model that accepts an array of yaw angles instead of this, and load the pretrained weights.
    max_episode_steps = 100

    map_size = (128, 128)



    case = 1
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    turbine_locations = read_turbine_positions(layout_file)

    env = TimeLimit(TurbineEnv(model, turbine_locations, render_mode="rgb_array", map_size=map_size[0]), max_episode_steps=max_episode_steps)

    wind_direction = np.array([225])
    yaws = np.array([225] * 10)
    options = {
        "wind_direction": wind_direction,
        "yaws": yaws
    }

    env.reset(options=options)

    env.render()

    print("Starting check", flush=True)
    env_checker.check_env(env)
