
import torch
from stable_baselines3 import TD3, SAC
from stable_baselines3.sac import MlpPolicy

from experiments.LESReinforcement.env_continuous import create_env
from utils.rl_utils import create_validation_points
from utils.sb3_callbacks import FigureRecorderCallback, TestComparisonCallback
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback

device = torch.device("cpu")


def train():
    case_nr = 1
    num_val_points = 100
    env = create_env()
    val_points = create_validation_points(case_nr, num_val_points, map_size=(128, 128))
    eval_callback = EveryNTimesteps(n_steps=1000, callback=TestComparisonCallback(env, val_points=val_points))

    fig_callback = FigureRecorderCallback(env)
    ntimestep_callback = EveryNTimesteps(n_steps=500, callback=fig_callback)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="sac_model")

    model = SAC(MlpPolicy, env, verbose=1, device=device, tensorboard_log="./sac_tensorboard/")
    model.learn(total_timesteps=500000, progress_bar=True, tb_log_name="SAC",
                callback=[checkpoint_callback, ntimestep_callback, eval_callback])
    model.save("SACTurbineEnvModel")

def predict():
    env = create_env()
    model = SAC.load("SACTurbineEnvModel")

    for i in range(10):
        obs, info = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, dones, truncations, info = env.step(action)
        env.render()


if __name__ == "__main__":
    train()
    predict()
