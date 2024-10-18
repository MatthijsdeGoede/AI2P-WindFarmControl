import torch
from stable_baselines3 import PPO

from experiments.LESReinforcement.env import create_env

device = torch.device("cpu")

def train():
    env = create_env()

    model = PPO("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log="./ppo_turbine_env/")
    model.learn(total_timesteps=200000, progress_bar=True, tb_log_name="first_run")
    model.save("TurbineEnvModel")

def predict():
    env = create_env()
    model = PPO.load("TurbineEnvModel")

    for i in range(10):
        obs, info = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, dones, truncations, info = env.step(action)
        env.render()


if __name__ == "__main__":
    train()
    predict()