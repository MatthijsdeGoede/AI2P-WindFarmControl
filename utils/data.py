import os
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


from numpy.linalg import norm
from scipy.interpolate import griddata


class MultiThread:
    def __init__(self, case, type):
        self.case = case
        self.type = type

    def compute(self, index):
        umean_abs, _, _ = vtk_to_umean_abs(f'./slices/{self.case}/{self.type}/{index}/U_slice_horizontal.vtk')
        np.save(f'./slices/{self.case}/Processed/{self.type}/Windspeed_map_scalars_{index}', umean_abs)
        print(f'Processed {index}')


def preprocess_vtk_files(case, type, overwrite=True):
    dirs = set(os.listdir(f'./slices/{case}/{type}'))
    os.makedirs(f'./slices/{case}/Processed/{type}', exist_ok=True)

    if not overwrite:
        existing = {file.split("_")[-1].split(".")[0] for file in os.listdir(f'./slices/{case}/Processed/{type}')}
        dirs = dirs - existing

    m = MultiThread(case, type)

    with Pool() as p:
        p.map(m.compute, dirs)


def import_vtk(file):
    """
    Imports standard SOWFA vtk files
    [data_type,cell_centers,cell_data] = import_vTK(file)

    input: file = location of vtk-file
    outputs:
    data_type = OpenFOAM label of measurement (e.g. U, Umean, have not tested for several measurements)
    cell_centers = locations of sampling (x,y,z)
    cell_data = sampling values (could be vectors (rows))
    """
    data_type = []
    cell_data = []

    with open(file, 'r') as f:
        lines = f.readlines()

    n_polygons = 0
    points_xyz = []
    polygons = []

    line_idx = 0
    while line_idx < len(lines):
        input_text = lines[line_idx].strip()

        if input_text == 'DATASET POLYDATA':
            line_idx += 1
            n_points = int(lines[line_idx].split()[1])
            points_xyz = np.array([list(map(float, lines[line_idx + i + 1].split())) for i in range(n_points)])
            line_idx += n_points

        elif input_text.startswith('POLYGONS'):
            n_polygons = int(input_text.split()[1])
            polygons = np.array([list(map(int, lines[line_idx + i + 1].split()))[1:] for i in range(n_polygons)])
            line_idx += n_polygons

        elif input_text.startswith('CELL_DATA'):
            n_attributes = int(lines[line_idx + 1].split()[-1])
            line_idx += 2
            for att in range(n_attributes):
                field_data = lines[line_idx].split()
                field_name = field_data[0]
                data_type.append(field_name)
                n_cells = int(field_data[2])

                if field_data[3] == 'float' and n_cells == n_polygons:
                    cell_data_att = np.array(
                        [list(map(float, lines[line_idx + i + 1].split())) for i in range(n_cells)])
                    cell_data.append(cell_data_att)
                    line_idx += n_cells
                else:
                    raise ValueError("Format problem")
                line_idx += 1

        line_idx += 1

    cell_data = np.concatenate(cell_data, axis=1) if len(cell_data) > 0 else None

    # Calculate cell centers
    if np.unique(polygons[:, 0]).shape[0] > 1:
        # Slow loop when the number of edges per cell is not constant
        cell_centers = np.zeros((n_polygons, 3))
        for i in range(n_polygons):
            tmp_cell_positions = points_xyz[polygons[i], :]
            cell_centers[i, :] = np.mean(tmp_cell_positions, axis=0)
    else:
        # Efficient calculation when the number of edges per cell is constant
        cell_centers = np.mean(points_xyz[polygons], axis=1)

    return data_type, cell_centers, cell_data


def vtk_to_umean_abs(file):
    """
    Imports standard SOWFA vtk files and calculates the interpolated mean absolute wind speed over the grid
    [umean_abs,x_axis,y_axis] = vtk_to_umean_abs(file)

    input: file = location of vtk-file
    outputs:
    u_mean_abs = interpolated mean absolute wind speed over the grid
    x_axis = x value range of the grid
    y_axis = y value range ofthe grid
    """
    # Import the vtk file
    data_type, cell_centers, cell_data = import_vtk(file)
    # Get the absolute mean velocity (magnitude of the vector)
    umean_abs_scattered = np.sqrt(np.sum(cell_data ** 2, axis=1))
    # Create a uniform 2D grid at hub-height
    x_axis = np.linspace(np.min(cell_centers[:, 0]), np.max(cell_centers[:, 0]), 300)
    y_axis = np.linspace(np.min(cell_centers[:, 1]), np.max(cell_centers[:, 1]), 300)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    # Interpolate the scattered data on the grid
    umean_abs = griddata(cell_centers[:, 0:2], umean_abs_scattered, (x_grid, y_grid), method='linear')
    return umean_abs, x_axis, y_axis


def get_data_from_file(type, i, case):
    return np.load(f'./slices/{case}/Processed/{type}/Windspeed_map_scalars_{i}.npy')


def angle_to_vec(wind_angle):
    angle_radians = np.deg2rad(wind_angle)
    return np.array([np.cos(angle_radians), np.sin(angle_radians)])


def read_wind_angles(file):
    return np.genfromtxt(file, delimiter=",")


def get_wind_vec_at_time(wind_angles, timestep):
    return angle_to_vec(wind_angles[wind_angles[:, 0] < timestep][-1, 1])


def get_wind_angles_for_range(file, custom_range, start_ts):
    wind_angles = read_wind_angles(file)
    return np.array([wind_angles[wind_angles[:, 0] < timestep - start_ts][-1, 1] for timestep in custom_range])


def read_turbine_positions(file):
    return np.genfromtxt(file, delimiter=",")[:, :2]


def get_angle_between_vec(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0)))


def create_turbine_nx_graph(pos, wind_vec, max_angle=90, max_dist=np.inf):
    G = nx.DiGraph()
    num_turbines = pos.shape[0]
    for i in range(num_turbines):
        G.add_node(i, pos=pos[i])

    for i in range(num_turbines):
        for j in range(num_turbines):
            if i != j:
                turb_vec = pos[j, :] - pos[i, :]
                angle = get_angle_between_vec(turb_vec, wind_vec)
                dist = norm(turb_vec)
                if dist < max_dist and angle < max_angle:
                    # add edge i to j
                    edge_feat = calculate_wake_distances(turb_vec, wind_vec, angle=angle)
                    G.add_edge(i, j, edge_feat=edge_feat)
    return G


def create_turbine_graph_tensors(pos, wind_vec, max_angle=90, max_dist=np.inf):
    num_turbines = pos.shape[0]
    src_nodes = []
    dst_nodes = []
    edge_feats = []
    for i in range(num_turbines):
        for j in range(num_turbines):
            if i != j:
                turb_vec = pos[j, :] - pos[i, :]
                angle = get_angle_between_vec(turb_vec, wind_vec)
                dist = norm(turb_vec)
                if dist < max_dist and angle < max_angle:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    feats = calculate_wake_distances(turb_vec, wind_vec, angle=angle)
                    edge_feats.append([feats[0], feats[1]])
    return torch.tensor([src_nodes, dst_nodes]), torch.tensor(edge_feats)


def calculate_wake_distances(turb_vec, wind_vec, angle=None):
    if angle is None:
        angle = get_angle_between_vec(turb_vec, wind_vec)
    angle_rad = np.deg2rad(angle)
    dist_i_j = norm(turb_vec)
    rad_dist = dist_i_j * np.sin(angle_rad)
    down_str_dist = dist_i_j * np.cos(angle_rad)
    return rad_dist, down_str_dist


def read_measurement(folder, measurement):
    dtypes = {
        "Turbine": int,
        "Time": float,
        "dt": float,
        "Feature": float
    }
    file_path = os.path.join(folder, measurement)
    df = pd.read_csv(file_path, sep=" ", header=1, names=["Turbine", "Time", "dt", "Measurement"], dtype=dtypes)
    df = df.loc[(df["Time"] % 5 == 0) & (df["Time"] > 30000), ["Turbine", "Time", "Measurement"]]
    df["Time"] = df["Time"].astype('int32')
    pivot_df = df.pivot(index="Turbine", columns="Time", values="Measurement")
    return pivot_df.to_numpy()


def prepare_graph_training_data():
    case_nr = 2
    wake_steering = False
    post_fix = "LuT2deg_internal" if wake_steering else "BL"
    start_ts = 30000
    min_ts = 30005
    max_ts = 42000
    step = 5
    data_range = range(min_ts, max_ts + 1, step)
    max_angle = 360

    data_dir = f"../../data/Case_0{case_nr}"
    flow_data_dir = f"{data_dir}/measurements_flow/postProcessing_{post_fix}"
    turbine_data_dir = f"{data_dir}/measurements_turbines/30000_{post_fix}"
    output_dir = f"{data_dir}/graphs/{post_fix}/{max_angle}"
    os.makedirs(output_dir, exist_ok=True)

    # layout_file = f"{data_dir}/HKN_12_to_15_layout_balanced.csv"
    # wind_angle_file = f"{data_dir}/HKN_12_to_15_dir.csv"

    layout_file = f"{data_dir}/HKN_06_to_09_layout_balanced.csv"
    wind_angle_file = f"{data_dir}/HKN_06_to_09_dir.csv"

    # Get the wind angles (global features) for every timestep in the simulation
    wind_angles = get_wind_angles_for_range(wind_angle_file, data_range, start_ts)  # (2400)

    # Get the features for every wind turbine (node features)
    turbine_pos = torch.tensor(read_turbine_positions(layout_file))  # (10x2)
    wind_speeds = torch.tensor(np.load(f"{flow_data_dir}/windspeed_estimation_case_0{case_nr}_30000_{post_fix}.npy")[0:, ::2][0:, step::step])  # (10x2400)
    yaw_measurement = torch.tensor(read_measurement(turbine_data_dir, "nacYaw"))  # (10x2400)
    rotation_measurement = torch.tensor(read_measurement(turbine_data_dir, "rotSpeed"))  # (10x2400)

    # Create custom dataset
    for i, timestep in enumerate(data_range):
        wind_vec = angle_to_vec(wind_angles[i])
        edge_index, edge_attr = create_turbine_graph_tensors(turbine_pos, wind_vec, max_angle=max_angle)
        # assert edge_index.size(1) == 90
        node_feats = torch.stack((wind_speeds[:, i], yaw_measurement[:, i], rotation_measurement[:, i]), dim=0).T
        target = torch.tensor(np.load(f"{flow_data_dir}/Windspeed_map_scalars/Windspeed_map_scalars_{timestep}.npy")).flatten()
        graph_data = Data(x=node_feats.float(), edge_index=edge_index, edge_attr=edge_attr.float(), y=target.float(), pos=turbine_pos)
        graph_data.global_feats = torch.tensor(wind_vec).reshape(-1, 2)
        # Save the graph with all data
        torch.save(graph_data, f"{output_dir}/graph_{timestep}.pt")


if __name__ == "__main__":
    prepare_graph_training_data()
    # wind_angles = read_wind_angles("../data/Case_01/HKN_12_to_15_dir.csv")
    # turbine_pos = read_turbine_positions("../data/Case_01/HKN_12_to_15_layout_balanced.csv")
    # timestep = 505
    # max_angle = 30
    # wind_vec = get_wind_vec_at_time(wind_angles, timestep)
    # graph = create_turbine_nx_graph(turbine_pos, wind_vec, max_angle=max_angle)
    # plot_graph(graph, wind_vec, max_angle=max_angle)

    # animate_mean_absolute_speed(30005)
    # umean_abs, x_axis, y_axis = vtk_to_umean_abs(
    #     '../data/Case_01/measurements_flow/postProcessing_BL/sliceDataInstantaneous/30505/U_slice_horizontal.vtk')
    # plot_mean_absolute_speed(umean_abs, x_axis, y_axis, graph, wind_vec)
    # animate_mean_absolute_speed(30005, comparison=True, case="Case_01")