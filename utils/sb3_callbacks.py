from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt

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