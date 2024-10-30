from utils.extract_windspeed import WindSpeedExtractor
from utils.preprocessing import read_turbine_positions
import numpy as np
import pandas as pd
from utils.rl_utils import create_validation_points
from utils.visualization import plot_mean_absolute_speed_subplot
from matplotlib import pyplot as plt
from stable_baselines3 import SAC, TD3, DDPG
from experiments.LESReinforcement.env_continuous import create_env


def validate_points(model, num_points=50):
    case_nr = 1
    seed = 42
    num_val_points = num_points
    env = create_env(render_mode="rgb_array")

    turbines = "12_to_15" if case_nr == 1 else "06_to_09" if case_nr == 2 else "00_to_03"
    layout_file = f"../../data/Case_0{case_nr}/HKN_{turbines}_layout_balanced.csv"
    turbine_locations = read_turbine_positions(layout_file)
    val_points = create_validation_points(case_nr, num_val_points, map_size=(128, 128), return_maps=True)
    avg_sim_greedy_power = np.mean([point['greedy_power'] for point in val_points])
    avg_sim_steering_power = np.mean([point['wake_power'] for point in val_points])

    greedy_val_power = []
    model_val_power = []

    wind_speed_extractor = WindSpeedExtractor(turbine_locations, 128)

    for i, val_point in enumerate(val_points):
        greedy_yaws = np.ones(10, dtype=float) * val_point["wind_direction"]
        greedy_actions = np.zeros(10, dtype=float)
        options = {"wind_direction": np.array([val_point["wind_direction"]]), "yaws": greedy_yaws}

        # Simulation greedy
        sim_greedy_turb_pixels = []
        wind_speed_greedy_sim = wind_speed_extractor(val_point["greedy_map"], val_point["wind_direction"],
                                                     val_point["greedy_yaws"], sim_greedy_turb_pixels)

        # Simulation wake steering
        sim_wake_turb_pixels = []
        wind_speed_extractor(val_point["wake_map"], val_point["wind_direction"], val_point["wake_yaws"],
                             sim_wake_turb_pixels)

        # Model greedy
        env.reset(seed=seed, options=options)
        _, rewards_greedy, _, _, info_greedy = env.step(greedy_actions)
        greedy_val_power.append(rewards_greedy)
        model_greedy_map, wind_vec, model_greedy_turb_pixels = env.get_render_info()

        # Model wake steering
        obs, info = env.reset(seed=seed, options=options)
        action, states = model.predict(obs)
        _, rewards_model, _, _, info_model = env.step(action)
        model_val_power.append(rewards_model)
        model_wake_map, _, model_wake_turb_pixels = env.get_render_info()

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        img1 = plot_mean_absolute_speed_subplot(axes[0, 0], val_point["greedy_map"], wind_vec, layout_file,
                                                sim_greedy_turb_pixels, color_bar=False)
        plot_mean_absolute_speed_subplot(axes[0, 1], model_greedy_map, wind_vec, layout_file, model_greedy_turb_pixels,
                                         color_bar=False)
        plot_mean_absolute_speed_subplot(axes[1, 0], val_point["wake_map"], wind_vec, layout_file, sim_wake_turb_pixels,
                                         color_bar=False)
        plot_mean_absolute_speed_subplot(axes[1, 1], model_wake_map, wind_vec, layout_file, model_wake_turb_pixels,
                                         color_bar=False)

        axes[0, 0].set_title(f"Simulation Greedy: {round(val_point['greedy_power'], 3)}")
        axes[0, 1].set_title(f"Model Greedy: {round(rewards_greedy, 3)}")
        axes[1, 0].set_title(f"Simulation Wake Steering: {round(val_point['wake_power'], 3)}")
        axes[1, 1].set_title(f"Model Wake Steering: {round(rewards_model, 3)}")
        axes[0, 0].set_aspect('equal', adjustable='box')
        axes[0, 1].set_aspect('equal', adjustable='box')
        axes[1, 0].set_aspect('equal', adjustable='box')
        axes[1, 1].set_aspect('equal', adjustable='box')

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(img1, cax=cbar_ax, orientation='vertical')
        plt.savefig(f'val_point_{i}.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    avg_model_greedy_power = np.mean(greedy_val_power)
    avg_model_steering_power = np.mean(model_val_power)

    print(f"sim greedy power: {avg_sim_greedy_power}")
    print(f"sim wake power: {avg_sim_steering_power}")
    print(f"model greedy power: {avg_model_greedy_power}")
    print(f"model wake power: {avg_model_steering_power}")


def plot_losses():
    # Load data from each CSV file
    ddpg_data = pd.read_csv('DDPG_1.csv')
    td3_data = pd.read_csv('TD3_1.csv')
    sac_data = pd.read_csv('SAC_2.csv')

    # Extract the training steps and mean average episode rewards
    # Assuming the columns are: [Index, Training Step, Mean Average Episode Reward]
    ddpg_steps = ddpg_data.iloc[:, 1]
    ddpg_rewards = ddpg_data.iloc[:, 2]

    td3_steps = td3_data.iloc[:, 1]
    td3_rewards = td3_data.iloc[:, 2]

    sac_steps = sac_data.iloc[:, 1]
    sac_rewards = sac_data.iloc[:, 2]

    # Plot each method's rewards against training steps
    plt.figure(figsize=(10, 6))
    plt.plot(ddpg_steps, ddpg_rewards, label='DDPG', color='blue')
    plt.plot(td3_steps, td3_rewards, label='TD3', color='green')
    plt.plot(sac_steps, sac_rewards, label='SAC', color='red')

    # Add labels and title
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Average Episode Reward')
    plt.title('Training Performance of DDPG, TD3, and SAC')
    plt.legend()

    # Show the plot
    plt.grid()
    plt.tight_layout()
    plt.savefig('rl_training.pdf', format='pdf', bbox_inches='tight')
    plt.show()


model = SAC.load("SACTurbineEnvModel.zip")
validate_points(model)
# plot_losses()