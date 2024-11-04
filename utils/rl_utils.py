import random
import torch
import numpy as np
from skimage.transform import resize

from utils.extract_windspeed import WindSpeedExtractor
from utils.preprocessing import read_turbine_positions, get_wind_angles_for_range, read_measurement


def create_validation_points(case_nr, num_points, seed=42, map_size=(128, 128), return_maps=False):
    """
    Creates a set of validation points based on the simulation input and output.

    Args:
        case_nr (int): The case number for the simulation (e.g., 1, 2).
        num_points (int): The number of validation points to generate.
        seed (int, optional): Random seed for reproducibility (default: 42).
        map_size (tuple, optional): Size of the wind speed maps (height, width), default is (128, 128).
        return_maps (bool, optional): If True, includes the wind speed maps in the output (default: False).

    Returns:
        list: A list of dictionaries containing wind direction, turbine yaws, and power outputs.
    """
    points = []
    random.seed(seed)
    data_dir = f"../../data/Case_0{case_nr}"

    # Directories for yaw measurements and wind speed maps
    greedy_yaw_dir = f"{data_dir}/measurements_turbines/30000_BL"
    wake_yaw_dir = f"{data_dir}/measurements_turbines/30000_LuT2deg_internal"
    greedy_map_dir = f"{data_dir}/measurements_flow/postProcessing_BL"
    wake_map_dir = f"{data_dir}/measurements_flow/postProcessing_LuT2deg_internal"

    # Select turbines based on the case number
    turbines = "12_to_15" if case_nr == 1 else "06_to_09" if case_nr == 2 else "00_to_03"
    wind_map_extractor = WindSpeedExtractor(read_turbine_positions(f"../../data/Case_0{case_nr}/HKN_{turbines}_layout_balanced.csv"), map_size[0])

    # Define the range of data points and retrieve wind angles
    data_range = range(30005, 42000 + 1, 5)
    wind_angles = get_wind_angles_for_range(f"{data_dir}/HKN_{turbines}_dir.csv", data_range, 30000)
    sample_range = list(enumerate(data_range))

    # Sample points from the data range
    samples = sample_range if num_points > len(data_range) else random.sample(sample_range, num_points)

    # Retrieve the yaws for both strategies
    all_greedy_yaws = (read_measurement(greedy_yaw_dir, "nacYaw") * -1 + 270) % 360
    all_wake_yaws = (read_measurement(wake_yaw_dir, "nacYaw") * -1 + 270) % 360

    # Iterate over selected timesteps to extract relevant data
    for i, timestep in samples:
        wind_angle = wind_angles[i]  # Retrieve the wind direction
        greedy_yaws = all_greedy_yaws[:, i].astype(int)  # Retrieve greedy yaws
        wake_yaws = all_wake_yaws[:, i].astype(int)  # Retrieve wake yaws

        # Load wind speed maps and reshape
        greedy_map = load_scalars(wake_map_dir, timestep, map_size).reshape(map_size[0], map_size[1])
        wake_map = load_scalars(greedy_map_dir, timestep, map_size).reshape(map_size[0], map_size[1])

        # Extract wind speed for both strategies
        greedy_wind_speed = wind_map_extractor(greedy_map, wind_angle, greedy_yaws)
        wake_wind_speed = wind_map_extractor(wake_map, wind_angle, wake_yaws)

        # Calculate power outputs for both strategies
        greedy_power = wind_speed_to_power(greedy_yaws, wind_angle, greedy_wind_speed)
        wake_power = wind_speed_to_power(wake_yaws, wind_angle, wake_wind_speed)

        if return_maps:
            # Include maps in the output if specified
            points.append({
                "wind_direction": wind_angle,
                "greedy_yaws": greedy_yaws,
                "greedy_map": greedy_map,
                "wake_yaws": wake_yaws,
                "wake_map": wake_map,
                "greedy_power": np.sum(greedy_power),
                "wake_power": np.sum(wake_power)
            })
        else:
            # Return only power outputs if maps are not requested
            points.append({
                "wind_direction": wind_angle,
                "greedy_power": np.sum(greedy_power),
                "wake_power": np.sum(wake_power)
            })

    return points


def wind_speed_to_power(yaws, wind_direction, wind_speed):
    """
    Converts wind speed and yaw angles to power output using the wind turbine power curve.

    Args:
        yaws (np.ndarray): Array of yaw angles for the turbines.
        wind_direction (float): Wind direction in degrees.
        wind_speed (np.ndarray): Array of wind speeds for the turbines.

    Returns:
        np.ndarray: Array of power outputs for each turbine based on the wind speed and yaw angles.
    """
    diff_yaw = np.deg2rad(yaws - wind_direction)  # Calculate the difference in yaw
    Pp = 2  # Power coefficient parameter
    Cp = np.cos(diff_yaw) ** Pp  # Calculate power coefficient based on yaw difference
    return (wind_speed ** 3) * Cp  # Calculate power output


def load_scalars(dir, timestep, map_size):
    """
    Loads wind speed scalar data from a specified directory and resizes it.

    Args:
        dir (str): Directory containing the wind speed scalar files.
        timestep (int): The timestep corresponding to the desired wind speed data.
        map_size (tuple): The desired size for the output map (height, width).

    Returns:
        torch.Tensor: A flattened tensor containing the resized wind speed data.
    """
    # Load the wind speed map and resize it
    return torch.tensor(resize(np.load(f"{dir}/windspeedMapScalars/Windspeed_map_scalars_{timestep}.npy"), map_size)).flatten()
