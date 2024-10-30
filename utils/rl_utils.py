import random
import torch
import numpy as np
from skimage.transform import resize

from utils.extract_windspeed import WindSpeedExtractor
from utils.preprocessing import read_turbine_positions, get_wind_angles_for_range, correct_angles, read_measurement
from utils.visualization import plot_prediction_vs_real


def create_validation_points(case_nr, num_points, seed=42, map_size=(128, 128), return_maps=False):
    points = []
    random.seed(seed)
    data_dir = f"../../data/Case_0{case_nr}"

    greedy_yaw_dir = f"{data_dir}/measurements_turbines/30000_BL"
    wake_yaw_dir = f"{data_dir}/measurements_turbines/30000_LuT2deg_internal"
    greedy_map_dir = f"{data_dir}/measurements_flow/postProcessing_BL"
    wake_map_dir = f"{data_dir}/measurements_flow/postProcessing_LuT2deg_internal"

    turbines = "12_to_15" if case_nr == 1 else "06_to_09" if case_nr == 2 else "00_to_03"
    wind_map_extractor = WindSpeedExtractor(read_turbine_positions(f"../../data/Case_0{case_nr}/HKN_{turbines}_layout_balanced.csv"), map_size[0])
    data_range = range(30005, 42000 + 1, 5)
    wind_angles = get_wind_angles_for_range(f"{data_dir}/HKN_{turbines}_dir.csv", data_range, 30000)
    sample_range = list(enumerate(data_range))
    if num_points > len(data_range):
        samples = sample_range
    else:
        samples = random.sample(sample_range, num_points)

    # Retrieve the yaws for both strategies
    all_greedy_yaws = (read_measurement(greedy_yaw_dir, "nacYaw") * -1 + 270) % 360
    all_wake_yaws = (read_measurement(wake_yaw_dir, "nacYaw") * -1 + 270) % 360

    for i, timestep in samples:
        # Retrieve the wind direction
        wind_angle = wind_angles[i]
        # Retrieve the scalar maps for both strategies
        greedy_yaws = all_greedy_yaws[:, i].astype(int)
        wake_yaws = all_wake_yaws[:, i].astype(int)
        greedy_map = load_scalars(wake_map_dir, timestep, map_size).reshape(map_size[0], map_size[1])
        wake_map = load_scalars(greedy_map_dir, timestep, map_size).reshape(map_size[0], map_size[1])

        # Extract the wind speed for both strategies
        greedy_wind_speed = wind_map_extractor(greedy_map, wind_angle, greedy_yaws)
        wake_wind_speed = wind_map_extractor(wake_map, wind_angle, wake_yaws)
        # Calculate the power for both strategies
        greedy_power = wind_speed_to_power(greedy_yaws, wind_angle, greedy_wind_speed)
        wake_power = wind_speed_to_power(wake_yaws, wind_angle, wake_wind_speed)

        if return_maps:
            # Return the angle, yaws and resulting maps
            points.append({"wind_direction": wind_angle, "greedy_yaws": greedy_yaws, "greedy_map": greedy_map,
                           "wake_yaws": wake_yaws, "wake_map": wake_map, "greedy_power": np.sum(greedy_power), "wake_power": np.sum(wake_power)})

        else:
            # Add the new validation point to the set
            points.append({"wind_direction": wind_angle, "greedy_power": np.sum(greedy_power), "wake_power": np.sum(wake_power)})
    return points


def wind_speed_to_power(yaws, wind_direction, wind_speed):
    diff_yaw = np.deg2rad(yaws - wind_direction)
    Pp = 2
    Cp = np.cos(diff_yaw) ** Pp
    return (wind_speed ** 3) * Cp #/ 512


def load_scalars(dir, timestep, map_size):
    return torch.tensor(resize(np.load(f"{dir}/windspeedMapScalars/Windspeed_map_scalars_{timestep}.npy"), map_size)).flatten()
