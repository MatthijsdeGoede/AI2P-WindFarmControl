import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback

from experiments.LESReinforcement.env import create_env
from utils.rl_utils import create_validation_points
from utils.sb3_callbacks import FigureRecorderCallback, TestComparisonCallback

device = torch.device("cpu")


def train():
    case_nr = 1
    num_val_points = 100
    env = create_env()
    val_points = create_validation_points(case_nr, num_val_points, map_size=(300, 300))
    eval_callback = EveryNTimesteps(n_steps=1000, callback=TestComparisonCallback(env, val_points=val_points))

    fig_callback = FigureRecorderCallback(env)
    ntimestep_callback = EveryNTimesteps(n_steps=500, callback=fig_callback)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="a2c_model")

    model = A2C("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log="./turbine_env/")
    model.learn(total_timesteps=200000, progress_bar=True, tb_log_name="A2C", callback=[checkpoint_callback, ntimestep_callback, eval_callback])
    model.save("A2CTurbineEnvModel")

def predict():
    env = create_env()
    model = A2C.load("A2CTurbineEnvModel")

    for i in range(10):
        obs, info = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, dones, truncations, info = env.step(action)
        env.render()


if __name__ == "__main__":
    train()
    predict()