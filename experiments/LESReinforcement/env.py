from typing import Optional

import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gymnasium import spaces
from gymnasium.utils import env_checker

import numpy as np

from architecture.pignn.deconv import DeConvNet
from architecture.pignn.pignn import FlowPIGNN
from experiments.graphs.graph_experiments import get_pignn_config
from utils.extract_windspeed import WindSpeedExtractor
from utils.preprocessing import read_turbine_positions, angle_to_vec, create_turbine_graph_tensors, correct_angles
from utils.rl_utils import wind_speed_to_power
from utils.visualization import plot_mean_absolute_speed, get_mean_absolute_speed_figure

device = torch.device("cpu")


class TurbineEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "matplotlib"], "render_fps": 4}

    def __init__(self, wind_speed_map_model, turbine_locations, render_mode=None, map_size=300, yaw_step=5, max_yaw=30):
        self.turbine_locations = torch.tensor(turbine_locations)
        self.n_turbines = len(turbine_locations)
        self.map_size = map_size

        # Model to predict the wind speed map
        self.model = wind_speed_map_model

        # Extract wind speeds from the map
        self.wind_speed_extractor = WindSpeedExtractor(turbine_locations, map_size)

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
        self._action_to_yaw = lambda action, wind: (action - self._n_yaw_steps) * yaw_step + wind
        self._yaw_to_action = lambda yaw, wind: (yaw - wind) // yaw_step + self._n_yaw_steps

        self._last_wind_speed = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            "wind_direction": self._wind_direction,
            "yaws": self._yaws,
        }

    def _get_info(self):
        return {
            "wind_speed": self._last_wind_speed
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is None:
            options = {}

        # If wind direction is given in options, use that
        if "wind_direction" in options:
            self._wind_direction = options["wind_direction"]
        else: # Else choose a wind direction that is in the data range
            self._wind_direction = correct_angles(self.np_random.integers(220, 261, size=(1,), dtype=int))

        # Similarly for yaws
        if "yaws" in options:
            self._yaws = self._yaw_to_action(options["yaws"], self._wind_direction[0])
        else:
            self._yaws = self.np_random.integers(0, 2 * self._n_yaw_steps - 1, size=self.n_turbines)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(action, self._wind_direction[0])

        # Use the graph model to create a wind speed map prediction
        wind_speed_map, _ = self.predict_wind_speed_map(yaws)

        self._last_wind_speed = self.wind_speed_extractor(wind_speed_map, self._wind_direction, yaws)

        # Convert the extracted wind speeds at the turbines to power
        power = wind_speed_to_power(yaws, self._wind_direction[0], self._last_wind_speed)

        # Take a step in the environment
        self._yaws = action
        observation = self._get_obs()
        info = self._get_info()

        # Return tuple in form observation, reward, terminated, truncated, info.
        # Power is the reward, and as of now, the environment does not terminate.
        return observation, np.sum(power), False, False, info

    def predict_wind_speed_map(self, yaws):
        x = torch.tensor(yaws).reshape(-1, 1).float()
        pos = self.turbine_locations
        wind_vec = angle_to_vec(self._wind_direction[0])
        ei, ef = create_turbine_graph_tensors(self.turbine_locations, wind_vec, max_angle=30)
        gf = torch.tensor(wind_vec).reshape(-1, 2)
        data = Data(x=x, edge_index=ei, edge_attr=ef.float(), pos=pos)
        data.global_feats = gf
        mini_batch = next(iter(DataLoader([data])))
        x, pos, edge_attr, glob = mini_batch.x, mini_batch.pos, mini_batch.edge_attr.float(), mini_batch.global_feats.float()
        nf = torch.cat((x, pos), dim=-1).float()

        self.model.eval()
        with torch.no_grad():
            wind_speed_map = self.model(mini_batch, nf, edge_attr, glob).reshape(self.map_size, self.map_size)

        return wind_speed_map, wind_vec

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "matplotlib":
            return self._render_frame()

    def _render_frame(self):
        # Render a wind speed map with current yaws
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(self._yaws, self._wind_direction[0])

        wind_speed_map, wind_vec = self.predict_wind_speed_map(yaws)

        turbine_pixels = []

        self.wind_speed_extractor(wind_speed_map, self._wind_direction, yaws, turbine_pixels)
        wind_vec = 75 * wind_vec

        if self.render_mode == "rgb_array":
            plot_mean_absolute_speed(wind_speed_map, wind_vec, windmill_blades=turbine_pixels)
        if self.render_mode == "matplotlib":
            return get_mean_absolute_speed_figure(wind_speed_map, wind_vec, windmill_blades=turbine_pixels)
        return


def create_env(case=1, max_episode_steps=100, render_mode="matplotlib", map_size=(300, 300)):
    # Make sure to actually use model that accepts an array of yaw angles instead of this, and load the pretrained weights.
    model_cfg = get_pignn_config()
    actor_model = DeConvNet(1, [64, 128, 256, 1], output_size=(map_size, map_size))
    model = FlowPIGNN(**model_cfg, actor_model=actor_model)
    model.load_state_dict(torch.load("model_case01/pignn_best.pt"))

    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    turbine_locations = read_turbine_positions(layout_file)

    env = TimeLimit(TurbineEnv(model, turbine_locations, render_mode=render_mode, map_size=map_size[0]), max_episode_steps=max_episode_steps)
    return env


if __name__ == "__main__":
    env = create_env()

    wind_direction = np.array([225])
    yaws = np.array([225] * 10)
    options = {
        "wind_direction": correct_angles(wind_direction),
        "yaws": correct_angles(yaws)
    }

    env.reset(options=options)

    env.render()

    print("Starting check", flush=True)
    env_checker.check_env(env.unwrapped)
