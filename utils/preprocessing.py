import os
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
import torch

from numpy.linalg import norm
from scipy.interpolate import griddata

from skimage.transform import resize

# from utils.timing import start_timer, print_timer


class MultiThread:
    def __init__(self, case, type):
        self.case = case
        self.type = type
        self.working_dir = f'../data/Case_0{self.case}/measurements_flow/postProcessing_{self.type}'

    def compute(self, index):
        umean_abs, _, _ = vtk_to_umean_abs(f'{self.working_dir}/sliceDataInstantaneous/{index}/U_slice_horizontal.vtk')
        np.save(f'{self.working_dir}/windspeedMapScalars/Windspeed_map_scalars_{index}', umean_abs)
        print(f'Processed {index}')


def preprocess_vtk_files(case, type, overwrite=True):
    working_dir = f'../data/Case_0{case}/measurements_flow/postProcessing_{type}/sliceDataInstantaneous'
    dirs = set(os.listdir(working_dir))

    if not overwrite:
        existing = {file.split("_")[-1].split(".")[0] for file in os.listdir(f'{working_dir}/windspeedMapScalars')}
        dirs = dirs - existing

    m = MultiThread(case, type)

    print(f'Processing in total {len(dirs)} files')

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


def read_wind_speed_scalars(type, i, case):
    return np.load(f'../data/{case}/measurements_flow/{type}/windspeedMapScalars/Windspeed_map_scalars_{i}.npy')


def angle_to_vec(wind_angle):
    angle_radians = np.deg2rad(wind_angle)
    return np.array([np.cos(angle_radians), np.sin(angle_radians)])


def correct_angles(angles):
    return (angles * -1 + 270) % 360


def read_wind_angles(file):
    angles = np.genfromtxt(file, delimiter=",") * np.array([1, -1]) + np.array([0, 270])
    return np.mod(angles, [np.inf, 360])


def get_wind_vec_at_time(wind_angles, timestep):
    return angle_to_vec(wind_angles[wind_angles[:, 0] <= timestep][-1, 1])


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


def resize_windspeed(windspeed_map, output_shape):
    return resize(windspeed_map, output_shape)


def get_yaws(case, type):
    yaws = read_measurement(f"../data/Case_0{case}/measurements_turbines/30000_{type}/", "nacYaw")

    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    wind_angles = read_wind_angles(f"../data/Case_0{case}/HKN_{turbines}_dir.csv")
    x_interp = np.linspace(5, 12000, yaws.shape[1], endpoint=True)
    y_interp = np.interp(x_interp, wind_angles[:, 0], wind_angles[:, 1])

    return (yaws * -1 + 270) - y_interp