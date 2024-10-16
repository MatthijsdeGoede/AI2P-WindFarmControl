from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import env_checker

import numpy as np

from architecture.windspeedLSTM.windspeedLSTM import WindspeedLSTM
from utils.extract_windspeed import WindspeedExtractor
from utils.preprocessing import read_turbine_positions


class TurbineEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, windspeedmap_model, turbine_locations, render_mode=None, map_size = 300, yaw_step=5, max_yaw=30):
        self.n_turbines = len(turbine_locations)

        # Model to predict the wind speed map
        self.model = windspeedmap_model

        # Extract wind speeds from the map
        self.windspeed_extractor = WindspeedExtractor(turbine_locations, map_size)

        self._n_yaw_steps = max_yaw // yaw_step

        # Define some variables to keep track off
        self._wind_direction = 0
        self._yaws = np.zeros(self.n_turbines, dtype=int)

        # Observations that are available to the agent, the current global wind direction and the current yaw angles
        self.observation_space = spaces.Dict(
            {
                "wind_direction": gym.spaces.Box(0, 360, dtype=int),
                "yaws": spaces.MultiDiscrete([2 * self._n_yaw_steps + 1] * self.n_turbines, start=[-self._n_yaw_steps] * self.n_turbines)
            }
        )

        # Actions the agents can take, it can select the new yaw angles for the turbines
        self.action_space = spaces.MultiDiscrete([2 * self._n_yaw_steps + 1] * self.n_turbines, start=[-self._n_yaw_steps] * self.n_turbines)

        # Convert action to an actual yaw angle
        self._action_to_yaw = lambda action : action * yaw_step
        self._yaw_to_action = lambda yaw : yaw // yaw_step

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
            self._wind_direction = options["wind_direction"]
        else:
            # Else generate a random direction (TODO: only use values actually observed in training the wind speed map)
            self._wind_direction = self.np_random.integers(0, 360, size=(1,), dtype=int)

        if "yaws" in options:
            self._yaws = options["yaws"]
        else:
            # TODO: check if this is allowed and does not mess with the RNG, else replace it with self.np_random
            self._yaws = self.np_random.integers(-self._n_yaw_steps, self._n_yaw_steps + 1, size=self.n_turbines)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(action)

        wind_speed_map = self.model(yaws)

        windspeeds = self.windspeed_extractor(wind_speed_map, self._wind_direction, yaws)

        # TODO convert windspeeds to a power estimate
        power = windspeeds

        self._yaws = self._yaw_to_action(yaws)

        observation = self._get_obs()
        info = self._get_info()

        # return tuple in form observation, reward, terminated, truncated, info.
        # Power is the reward, and as of now, the environment does not terminate.
        return observation, power, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Render a wind speed map with current yaws

        return

if __name__ == "__main__":
    # Make sure to actually use model that accepts an array of yaw angles instead of this, and load the pretrained weights.
    model = WindspeedLSTM(50)

    case = 1
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    turbine_locations = read_turbine_positions(layout_file)

    env = TurbineEnv(model, turbine_locations)
    env_checker.check_env(env)