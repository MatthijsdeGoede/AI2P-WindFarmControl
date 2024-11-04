import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy

from experiments.reinforcement_learning.env_continuous import create_env
from utils.rl_utils import create_validation_points
from utils.sb3_callbacks import FigureRecorderCallback, TestComparisonCallback
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback

device = torch.device("cpu")


def train():
    """
    Trains a DDPG model on a continuous turbine environment.

    This function initializes the environment, sets up validation points,
    and configures various callbacks for training. It creates a DDPG model
    with an Ornstein-Uhlenbeck noise process to encourage exploration and
    begins the training process.

    The training model is saved after completion, along with periodic
    checkpoints and visualizations.

    Returns:
        None
    """
    case_nr = 1
    num_val_points = 100

    # Create the continuous turbine environment
    env = create_env()

    # Generate validation points for evaluation during training
    val_points = create_validation_points(case_nr, num_val_points, map_size=(128, 128))

    # Set up evaluation callback to validate model performance every 1000 steps
    eval_callback = EveryNTimesteps(n_steps=1000, callback=TestComparisonCallback(env, val_points=val_points))

    # Set up callback to record figures
    fig_callback = FigureRecorderCallback(env)
    ntimestep_callback = EveryNTimesteps(n_steps=500, callback=fig_callback)

    # Set up checkpoint callback to save the model every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="ddpg_model")

    # Initialize the noise object for the DDPG agent
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # Create the DDPG model
    model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1, device=device, tensorboard_log="./tensorboard/")

    # Start training the model
    model.learn(total_timesteps=300000, progress_bar=True, tb_log_name="DDPG",
                callback=[checkpoint_callback, ntimestep_callback, eval_callback])

    # Save the trained model
    model.save("DDPGTurbineEnvModel")


def predict():
    """
    Runs a prediction using the trained DDPG model on the continuous turbine environment.

    This function loads the previously trained DDPG model and uses it to
    perform predictions in the environment. It resets the environment and
    takes actions based on the model's predictions for 10 episodes, rendering
    the environment's state after each action.

    Returns:
        None
    """
    # Create the continuous turbine environment
    env = create_env()

    # Load the trained DDPG model
    model = DDPG.load("DDPGTurbineEnvModel")

    # Run the prediction for 10 iterations
    for i in range(10):
        # Reset the environment and get the initial observation
        obs, info = env.reset()

        # Predict the action using the model
        action, _states = model.predict(obs)

        # Step through the environment with the predicted action
        obs, rewards, dones, truncations, info = env.step(action)

        # Render the environment
        env.render()


if __name__ == "__main__":
    train()
    # predict()
