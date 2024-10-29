import os
import sys
sys.path.append(os.getcwd())

from typing import Optional

import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FlattenObservation
from gymnasium import spaces
from gymnasium.utils import env_checker

import numpy as np

from experiments.NavierStokes.PINN import PINN

from utils.extract_windspeed import WindSpeedExtractor
from utils.preprocessing import read_turbine_positions, angle_to_vec, create_turbine_graph_tensors, correct_angles
from utils.rl_utils import wind_speed_to_power
from utils.visualization import plot_mean_absolute_speed, get_mean_absolute_speed_figure

device = torch.device("cuda")

HUB_HEIGHT = 119 #Height of the turbine in meters
TOTAL_SIMULATION_SECONDS = 12000 #Total duration of the LES simulation the PINN was trained on
TIME_SCALING_FACTOR = 1/6000 #One second delta in real life corresponds to a 1/6000 delta on the model input

class ContinuousTurbineEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "matplotlib"], "render_fps": 4}

    def __init__(self, wind_speed_map_model, turbine_locations, render_mode=None, map_size=300, max_yaw=30, pinn_time_start_point=0.2, dynamic_time=False):
        self.turbine_locations = torch.tensor(turbine_locations)
        self.n_turbines = len(turbine_locations)
        self.map_size = map_size
        self.current_step = 0

        self.pinn_start_time = (TOTAL_SIMULATION_SECONDS * pinn_time_start_point * TIME_SCALING_FACTOR) - 1
        self.dynamic_time = False

        # Model to predict the wind speed map
        self.model = wind_speed_map_model

        # Extract wind speeds from the map
        self.wind_speed_extractor = WindSpeedExtractor(turbine_locations, map_size)

        # Define some variables to keep track off
        self.wind_direction = [0]
        self.yaws = np.zeros(self.n_turbines, dtype=float)

        # Observations that are available to the agent, the current global wind direction and the current yaw angles
        self.observation_space = spaces.Dict(
            {
                "wind_direction": gym.spaces.Box(0, 360, dtype=np.float64),
                "yaws": gym.spaces.Box(low=np.full((10), -1), high=np.full((10), 1), dtype=float)
            }
        )

        # Actions the agents can take, it can select the new yaw angles for the turbines
        self.action_space = gym.spaces.Box(low=np.full((10), -1), high=np.full((10), 1), dtype=float)

        # Convert action to an actual yaw angle
        self._action_to_yaw = lambda action, wind: action * max_yaw + wind
        self._yaw_to_action = lambda yaw, wind: (yaw - wind) / max_yaw

        self._last_wind_speed = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @property
    def yaws(self):
        return self._yaws
    
    @yaws.setter
    def yaws(self, value):
        #print(f"New Yaws: {value=}")
        self._yaws = value

    @property
    def wind_direction(self):
        return self._wind_direction
    
    @wind_direction.setter
    def wind_direction(self, value):
        #print(f"New Wind Direction: {value=}")
        self._wind_direction = np.array(value).astype(np.float64) 

    def _get_obs(self):
        return {
            "wind_direction": self.wind_direction,
            "yaws": self.yaws,
        }

    def _get_info(self):
        return {
            "wind_speed": self._last_wind_speed
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_step = 0

        if options is None:
            options = {}

        # If wind direction is given in options, use that
        if "wind_direction" in options:
            self.wind_direction = options["wind_direction"]
        else: # Else choose a wind direction that is in the data range
            self.wind_direction = self.np_random.integers(220, 261, size=(1,))

        # Similarly for yaws
        if "yaws" in options:
            self.yaws = self._yaw_to_action(options["yaws"], self.wind_direction[0])
        else:
            self.yaws = self.np_random.uniform(-1, 1, size=self.n_turbines)

        observation = self._get_obs()
        info = self._get_info()

        #print(f"{observation=} \n{info=}")
        #print("------------------- \n")

        return observation, info

    def step(self, action):
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(action, self.wind_direction[0])

        wind_speed_map, _ = self.predict_wind_speed_map(yaws, self.wind_direction[0])
        self._last_wind_speed = self.wind_speed_extractor(wind_speed_map, yaws)

        # Convert the extracted wind speeds at the turbines to power
        power = wind_speed_to_power(yaws, self.wind_direction[0], self._last_wind_speed)

        # Take a step in the environment
        #print(f"{action=}")
        self.yaws = action
        observation = self._get_obs()
        info = self._get_info()

        # Return tuple in form observation, reward, terminated, truncated, info.
        # Power is the reward, and as of now, the environment does not terminate.
        return observation, np.mean(power), False, False, info

    def predict_wind_speed_map(self, yaws, wind_direction):
        yaw_tensor = torch.from_numpy(yaws)
        wind_tensor = torch.tensor([wind_direction])
        
        # print(f"{yaw_tensor=} \n{wind_tensor=}")
        pinn_input = self._prepare_pinn_input(yaw_tensor, wind_tensor).float().cuda(non_blocking=True)

        with torch.no_grad():
            wind_speed_map = self.model(pinn_input)[0][:, [0, 1]] * 8
        
        #Calculating the magnitude of the flow using x and y components
        wind_speed_map = torch.sqrt(wind_speed_map[:, 0]**2 + wind_speed_map[:, 1]**2)
        wind_speed_map = wind_speed_map.reshape(300, 300).detach().cpu()

        return wind_speed_map, pinn_input[0, -2:].detach().cpu()

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "matplotlib":
            return self._render_frame()

    def _render_frame(self):
        # Render a wind speed map with current yaws
        # Convert actions to actual yaw values
        yaws = self._action_to_yaw(self.yaws, self.wind_direction[0])

        wind_speed_map, wind_vec = self.predict_wind_speed_map(yaws, self.wind_direction[0])

        turbine_pixels = []

        self.wind_speed_extractor(wind_speed_map, correct_angles(yaws), turbine_pixels)
        wind_vec = 75 * wind_vec

        if self.render_mode == "rgb_array":
            plot_mean_absolute_speed(wind_speed_map, wind_vec, windmill_blades=turbine_pixels)
        if self.render_mode == "matplotlib":
            return get_mean_absolute_speed_figure(wind_speed_map, torch.tensor([-wind_vec[0], -wind_vec[1]]), windmill_blades=turbine_pixels)
        return

    def _convert_wind_vec_to_degrees(self, sin_val, cos_val):
        angle_radians = torch.arctan2(sin_val, cos_val)
        angle_degrees = torch.rad2deg(angle_radians)

        if (angle_degrees < 0):
            angle_degrees += 360

        return angle_degrees

    def _convert_degrees_to_wind_vec(self, degree):
        radian = torch.deg2rad(degree)
        out = torch.stack((torch.sin(radian), torch.cos(radian))).T
        return out
    
    def _prepare_pinn_input(self, turbine_yaws, wind_direction):
        linspace = torch.linspace(-0.5, 0.5, 300)
        x_coords = linspace.repeat(300)
        y_coords = linspace.repeat_interleave(300)
        z_coords = torch.full((90000,), (HUB_HEIGHT / 5000) - 0.5)

        if (self.dynamic_time):
            #Let's assume each step in RL corresponds to 5 seconds
            time = self.pinn_start_time + (self.current_step * TIME_SCALING_FACTOR * 5)
            assert time < 1

        else:
            time = 0

        time_coords = torch.full((90000,), time)


        coords = torch.stack((x_coords, y_coords, z_coords, time_coords)).T

        yaw_vec = self._convert_degrees_to_wind_vec(turbine_yaws).reshape(1, -1)
        wind_vec = self._convert_degrees_to_wind_vec(wind_direction).reshape(1, -1)

        yaw_wind_vec = torch.concat((yaw_vec, wind_vec), dim=1)
        yaw_wind_vec = yaw_wind_vec.repeat(90000, 1)
        
        pinn_input = torch.concat((coords, yaw_wind_vec), dim=1)



        return pinn_input



def create_env(max_episode_steps=100, max_yaw=30, render_mode="matplotlib", pinn_start=0.2, dynamic_time=False):
    checkpoint = torch.load("experiments/NavierStokes/models/SiLU256-d5-p0.01-f/model620.pt")
    model = PINN(in_dimensions=26, hidden_size=256, out_dimensions=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()

    turbine_locations = read_turbine_positions(f"data/Case_01/HKN_12_to_15_layout_balanced.csv")

    env = FlattenObservation(TimeLimit(ContinuousTurbineEnv(model, 
                                                        turbine_locations, 
                                                        render_mode=render_mode,
                                                        map_size=300, 
                                                        max_yaw=max_yaw, 
                                                        pinn_time_start_point=pinn_start, 
                                                        dynamic_time=dynamic_time), 
                                max_episode_steps=max_episode_steps))

    return env


if __name__ == "__main__":
    env = create_env()

    wind_direction = np.array([225])
    yaws = np.array([225.0] * 10, dtype=float)

    env.reset()
    env.render()

    #print("Starting check", flush=True)
    env_checker.check_env(env.unwrapped)
