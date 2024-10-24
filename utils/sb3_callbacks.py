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
        greedy = np.ones(10) * 7  # Actions is a discrete space, where 7 is the middle and thus 0 degrees yaw
        _, rewards_greedy, _, _, info_greedy = self.eval_env.step(greedy)
        fig_greedy = self.eval_env.render()
        if fig_greedy:
            self.logger.record("evaluation/greedy", Figure(fig_greedy, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

        # Model control
        obs, info = self.eval_env.reset(seed=seed)
        action, states = self.model.predict(obs)
        _, rewards_model, _, _, info_model = self.eval_env.step(action)
        fig_model = self.eval_env.render()
        if fig_model:
            self.logger.record("trajectory/model", Figure(fig_model, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

        self.logger.record("evaluation/greedy_wind_speeds", np.mean(info_greedy['wind_speed']))
        self.logger.record("evaluation/model_wind_speeds", np.mean(info_model['wind_speed']))
        self.logger.record("evaluation/min_greedy_wind_speeds", np.min(info_greedy['wind_speed']))
        self.logger.record("evaluation/min_model_wind_speeds", np.min(info_model['wind_speed']))
        self.logger.record("evaluation/max_greedy_wind_speeds", np.max(info_greedy['wind_speed']))
        self.logger.record("evaluation/max_model_wind_speeds", np.max(info_model['wind_speed']))

        self.logger.record("evaluation/greedy_reward", rewards_greedy)
        self.logger.record("evaluation/model_reward", rewards_model)
        print(f"rewards_greedy: {rewards_greedy}, rewards_model: {rewards_model}")
        print(f"info_greedy: {info_greedy['wind_speed']}, info_model: {info_model['wind_speed']}")
        return True


class TestComparisonCallback(BaseCallback):
    def __init__(self, eval_env: Union[Env, VecEnv], val_points, verbose=0):
        super().__init__(verbose)
        self.val_points = val_points
        self.avg_sim_greedy_power = np.avg([point['greedy_power'] for point in val_points])
        self.avg_sim_steering_power = np.avg([point['wake_power'] for point in val_points])
        self.eval_env = eval_env

    def _on_step(self):
        seed = 42
        greedy_val_power = []
        model_val_power = []
        for val_point in self.val_points:
            greedy_yaws = np.ones(10) * 7
            options = {"wind_direction": val_point["wind_direction"], "yaws": greedy_yaws}

            # Model greedy
            self.eval_env.reset(seed=seed, options=options)
            _, rewards_greedy, _, _, info_greedy = self.eval_env.step(greedy_yaws)
            greedy_val_power.append(rewards_greedy)

            # Model wake steering
            obs, info = self.eval_env.reset(seed=seed, options=options)
            action, states = self.model.predict(obs)
            _, rewards_model, _, _, info_model = self.eval_env.step(action)
            model_val_power.append(rewards_model)

        mean_greedy_power = np.mean(greedy_val_power)
        mean_steering_power = np.mean(model_val_power)
        self.logger.record("evaluation/avg_model_greedy_power", mean_greedy_power)
        self.logger.record("evaluation/avg_model_steering_power", mean_steering_power)
        self.logger.record("evaluation/avg_sim_greedy_power", self.avg_sim_greedy_power)
        self.logger.record("evaluation/avg_sim_steering_power", self.avg_sim_steering_power)

        print(f"avg greedy power (model, sim): ({mean_greedy_power}, {self.avg_sim_greedy_power})")
        print(f"avg steering power (model, sim): {mean_steering_power}, {self.avg_sim_steering_power}")
        return True
