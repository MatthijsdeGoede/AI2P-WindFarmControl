import os
import sys
sys.path.append(os.getcwd())
import argparse

import torch
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback

from experiments.LESReinforcement.env_cont_pinn import create_env

from utils.rl_utils import create_validation_points
from utils.sb3_callbacks import FigureRecorderCallback, TestComparisonCallback

device = torch.device("cuda")

def train(max_steps, max_yaw, dynamic_time, pinn_start):
    env = create_env(max_episode_steps=max_steps, max_yaw=max_yaw, dynamic_time=dynamic_time, pinn_start=pinn_start)

    val_points = create_validation_points(case_nr=1, num_points=100, map_size=(300, 300))
    eval_callback = EveryNTimesteps(n_steps=250, callback=TestComparisonCallback(env, val_points=val_points))

    fig_callback = FigureRecorderCallback(env)
    ntimestep_callback = EveryNTimesteps(n_steps=250, callback=fig_callback)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="td3_model")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log="./turbine_env/")

    log_path = f"TD3-m{max_steps}-y{max_yaw}-{dynamic_time}{f'-{pinn_start}' if dynamic_time else ''}"
    model.learn(total_timesteps=100000, progress_bar=True, tb_log_name=log_path,
                callback=[checkpoint_callback, ntimestep_callback, eval_callback])
    
    model.save("TD3TurbineEnvModel")


def predict(max_steps, max_yaw, dynamic_time, pinn_start):
    env = create_env(max_episode_steps=max_steps, max_yaw=max_yaw, dynamic_time=dynamic_time, pinn_start=pinn_start)
    model = TD3.load("TD3TurbineEnvModel")

    for i in range(10):
        obs, info = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, dones, truncations, info = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your RL model")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-yaw", type=int, default=10)
    parser.add_argument("--d-time", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pinn-start", type=float, default=0.2)

    args = parser.parse_args()

    train(args.max_steps, args.max_yaw, args.d_time, args.pinn_start)
    predict(args.max_steps, args.max_yaw, args.d_time, args.pinn_start)
