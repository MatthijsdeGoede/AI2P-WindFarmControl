from typing import Union

import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnv


class FigureRecorderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self):
        fig = self.env.render()
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

class ComparisonCallback(BaseCallback):
    def __init__(self, eval_env: Union[Env, VecEnv], verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self):
        seed = np.random.randint(10000)
        self.eval_env.reset(seed=seed)
        greedy = np.ones(10) * 7 # Actions is a discrete space, where 7 is the middle and thus 0 degrees yaw
        _, rewards_greedy, _, _, info_greedy = self.eval_env.step(greedy)
        self.eval_env.render()

        # Model control
        obs, info = self.eval_env.reset(seed=seed)
        action, states = self.model.predict(obs)
        _, rewards_model, _, _, info_model = self.eval_env.step(action)
        self.eval_env.render()

        self.logger.record("evaluation/greedy_windspeeds", np.mean(info_greedy['windspeed']))
        self.logger.record("evaluation/model_windspeeds", np.mean(info_model['windspeed']))
        self.logger.record("evaluation/min_greedy_windspeeds", np.min(info_greedy['windspeed']))
        self.logger.record("evaluation/min_model_windspeeds", np.min(info_model['windspeed']))
        self.logger.record("evaluation/max_greedy_windspeeds", np.max(info_greedy['windspeed']))
        self.logger.record("evaluation/max_model_windspeeds", np.max(info_model['windspeed']))

        self.logger.record("evaluation/greedy_reward",rewards_greedy)
        self.logger.record("evaluation/model_reward", rewards_model)
        print(f"rewards_greedy: {rewards_greedy}, rewards_model: {rewards_model}")
        print(f"info_greedy: {info_greedy['windspeed']}, info_model: {info_model['windspeed']}")
        return True


