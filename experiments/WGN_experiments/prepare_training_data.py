import os

import numpy as np
import torch

from utils.preprocessing import get_wind_angles_for_range, read_turbine_positions, read_measurement, angle_to_vec, \
    create_turbine_graph_tensors
from torch_geometric.data import Data

from skimage.transform import resize


def prepare_graph_training_data():
    case_nr = 3
    wake_steering = True
    type = "LuT2deg_internal" if wake_steering else "BL"
    start_ts = 30000
    min_ts = 30005
    max_ts = 41995
    step = 5
    data_range = range(min_ts, max_ts + 1, step)
    max_angle = 30

    data_dir = f"../../Data/target_data_resized/Case_0{case_nr}"
    flow_data_dir = f"{data_dir}/postProcessing_{type}"
    turbine_data_dir = f"../../Data/input_data/Case_0{case_nr}"
    turbines = "12_to_15" if case_nr == 1 else "06_to_09" if case_nr == 2 else "00_to_03"
    output_dir = f"../../Data/WGN_train_data"

    os.makedirs(output_dir, exist_ok=True)

    layout_file = f"{turbine_data_dir}/HKN_{turbines}_layout_balanced.csv"
    wind_angle_file = f"{turbine_data_dir}/HKN_{turbines}_dir.csv"

    # Get the wind angles (global features) for every timestep in the simulation
    wind_angles = get_wind_angles_for_range(wind_angle_file, data_range, start_ts)  # (2400)

    # Get the features for every wind turbine (node features)
    turbine_pos = torch.tensor(read_turbine_positions(layout_file))  # (10x2)
    wind_speeds = torch.tensor(np.load(f"{turbine_data_dir}/turbine_measurements/windspeed_estimation_case_0{case_nr}_30000_{type}.npy")[0:, ::2][0:, step::step])  # (10x2400)
    yaw_measurement = (torch.tensor(read_measurement(f"{turbine_data_dir}/turbine_measurements/30000_{type}", "nacYaw")) * -1 + 270) % 360  # (10x2400)


    # Create custom dataset
    for i, timestep in enumerate(data_range):
        wind_vec = angle_to_vec(wind_angles[i])
        edge_index, edge_attr = create_turbine_graph_tensors(turbine_pos, wind_vec, max_angle=max_angle)
        # assert edge_index.size(1) == 90
        node_feats = torch.stack((wind_speeds[:, i], yaw_measurement[:, i]), dim=0).T
        # node_feats = yaw_measurement[:, i].reshape(-1, 1)

        target = torch.tensor(resize(np.load(f"{flow_data_dir}/Windspeed_map_scalars_{timestep}.npy"), (128, 128))).flatten()
        graph_data = Data(x=node_feats.float(), edge_index=edge_index, edge_attr=edge_attr.float(), y=target.float(), pos=turbine_pos)
        graph_data.global_feats = torch.tensor(wind_vec).reshape(-1, 2)
        # Save the graph with all data
        torch.save(graph_data, f"{output_dir}/graph_{case_nr}_{type}_{timestep}.pt")


if __name__ == "__main__":
    prepare_graph_training_data()