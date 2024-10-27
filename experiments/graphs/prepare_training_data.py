import os

import numpy as np
import torch

from utils.preprocessing import get_wind_angles_for_range, read_turbine_positions, read_measurement, angle_to_vec, \
    create_turbine_graph_tensors
from torch_geometric.data import Data

from skimage.transform import resize


def prepare_graph_training_data(case_nr=1, wake_steering=False, max_angle=30):
    map_size = 128
    type = "LuT2deg_internal" if wake_steering else "BL"
    start_ts = 30000
    min_ts = 30005
    max_ts = 42000
    step = 5
    data_range = range(min_ts, max_ts + 1, step)

    data_dir = f"../../data/Case_0{case_nr}"
    flow_data_dir = f"{data_dir}/measurements_flow/postProcessing_{type}"
    turbine_data_dir = f"{data_dir}/measurements_turbines/30000_{type}"
    turbines = "12_to_15" if case_nr == 1 else "06_to_09" if case_nr == 2 else "00_to_03"
    output_dir = f"{data_dir}/graphs/{type}/{max_angle}"
    os.makedirs(output_dir, exist_ok=True)

    layout_file = f"{data_dir}/HKN_{turbines}_layout_balanced.csv"
    wind_angle_file = f"{data_dir}/HKN_{turbines}_dir.csv"

    # Get the wind angles (global features) for every timestep in the simulation
    wind_angles = get_wind_angles_for_range(wind_angle_file, data_range, start_ts)  # (2400)

    # Get the features for every wind turbine (node features)
    turbine_pos = torch.tensor(read_turbine_positions(layout_file))  # (10x2)
    yaw_measurement = (torch.tensor(read_measurement(turbine_data_dir, "nacYaw")) * -1 + 270) % 360  # (10x2400)

    # wind_speeds = torch.tensor(np.load(f"{flow_data_dir}/windspeed_estimation_case_0{case_nr}_30000_{type}.npy")[0:, ::2][0:, step::step])  # (10x2400)
    # rotation_measurement = torch.tensor(read_measurement(turbine_data_dir, "rotSpeed"))  # (10x2400)

    for i, timestep in enumerate(data_range):
        wind_vec = angle_to_vec(wind_angles[i])
        edge_index, edge_attr = create_turbine_graph_tensors(turbine_pos, wind_vec, max_angle=max_angle)
        # node_feats = torch.stack((wind_speeds[:, i], yaw_measurement[:, i], rotation_measurement[:, i]), dim=0).T
        node_feats = yaw_measurement[:, i].reshape(-1, 1)
        target = torch.tensor(resize(np.load(f"{flow_data_dir}/windspeedMapScalars/Windspeed_map_scalars_{timestep}.npy"), (map_size, map_size))).flatten()
        graph_data = Data(x=node_feats.float(), edge_index=edge_index, edge_attr=edge_attr.float(), y=target.float(), pos=turbine_pos)
        graph_data.global_feats = torch.tensor(wind_vec).reshape(-1, 2)
        torch.save(graph_data, f"{output_dir}/graph_{timestep}.pt")
    print(f"Prepared training data for case: {case_nr}, wake steering: {wake_steering} and max angle: {max_angle}")


if __name__ == "__main__":
    prepare_graph_training_data(1, False, 30)
    prepare_graph_training_data(2, False, 30)
    prepare_graph_training_data(3, False, 30)

    prepare_graph_training_data(1, True, 30)
    prepare_graph_training_data(2, True, 30)
    prepare_graph_training_data(3, True, 30)

    prepare_graph_training_data(1, False, 90)
    prepare_graph_training_data(2, False, 90)
    prepare_graph_training_data(3, False, 90)

    prepare_graph_training_data(1, True, 90)
    prepare_graph_training_data(2, True, 90)
    prepare_graph_training_data(3, True, 90)

    prepare_graph_training_data(1, False, 360)
    prepare_graph_training_data(2, False, 360)
    prepare_graph_training_data(3, False, 360)

    prepare_graph_training_data(1, True, 360)
    prepare_graph_training_data(2, True, 360)
    prepare_graph_training_data(3, True, 360)
