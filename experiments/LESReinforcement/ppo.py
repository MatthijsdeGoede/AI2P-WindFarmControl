import gymnasium as gym

import stable_baselines3
import torch
from stable_baselines3 import PPO

from architecture.pignn.deconv import DeConvNet
from architecture.pignn.pignn import FlowPIGNN
from experiments.LESReinforcement.env import TurbineEnv
from experiments.graphs.graph_experiments import get_pignn_config
from utils.preprocessing import read_turbine_positions

device = torch.device("cpu")

# Parallel environments
# Make sure to actually use model that accepts an array of yaw angles instead of this, and load the pretrained weights.
model_cfg = get_pignn_config()
actor_model = DeConvNet(1, [128, 256, 1]).to(device)
flow_model = FlowPIGNN(**model_cfg, actor_model=actor_model).to(device)
flow_model.load_state_dict(torch.load("model_case01/pignn_best.pt"))

case = 1
turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
layout_file = f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
turbine_locations = read_turbine_positions(layout_file)

env = TurbineEnv(flow_model, turbine_locations, render_mode="rgb_array")

# model = PPO("MultiInputPolicy", env, verbose=1, device=device)
# model.learn(total_timesteps=2000, progress_bar=True)
# model.save("TurbineEnvModel")

model = PPO.load("TurbineEnvModel")

obs, info = env.reset()
action, _states = model.predict(obs)
obs, rewards, dones, truncations, info = env.step(action)
