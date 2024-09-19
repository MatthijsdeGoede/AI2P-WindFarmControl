import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Circle
from numpy.linalg import norm
from scipy.interpolate import griddata


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


def animate_mean_absolute_speed(start, frames=None, comparison=False, case="Case_1"):
    """
    Creates an animation of one or two wind speed maps over time, in 5 second increments. Uses the preprocessed data.
    Data is expected to be located in ./slices/Processed/BL/*_xxxxx.npy files, and contain the precomputed mean
    absolute wind speed. xxxxx denotes the timestamp of the wind speed measurement in seconds.
    @param start: Start timestamp to render from.
    @param frames: Amount of frames to render, leave empty to render till the end.
    @param comparison: Change to True if you want to create two plots, one with wake steering and one without.
    """
    if frames is None:
        dirs = os.listdir(f'./slices/{case}/Processed')
        all_files = [os.listdir(f'./slices/{case}/Processed/{dir}') for dir in dirs]
        biggest_file = [int(max(files, key=lambda x: int(x.split('.')[0].split('_')[-1])).split('.')[0].split('_')[-1]) for files in all_files]
        frames = (min(biggest_file) - start)//5 + 1
        print(frames)

    images = []

    def setup_image(ax, type):
        umean = get_data_from_file(type, start, case)
        axes_image = ax.imshow(umean, animated=True)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Distance (m)")
        images.append((axes_image, type))

    fig, (ax1) = plt.subplots(1, 1)

    setup_image(ax1, "BL")

    # setup_image(ax2, "LuT2deg")
    # fig.colorbar(axes_image_1)
    if comparison:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        setup_image(ax1, "BL")
        ax1.set_title("Greedy Controller")
        setup_image(ax2, "LuT2deg")
        ax2.set_title("Wake Steering")
    else:
        fig, (ax1) = plt.subplots(1, 1)
        setup_image(ax1, "BL")
        fig.colorbar(images[0][0], ax=ax1)

    def animate(i):
        # fig.suptitle(f"Interpolated UmeanAbs at Hub-Height\nRuntime of simulation: {5 * i} seconds")

        for axes_image, type in images:
            umean_abs = get_data_from_file(type, start + 5 * i, case)
            axes_image.set_data(umean_abs)
        return fig, *images

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=frames, interval=50)
    os.makedirs(f'./animations/{case}/{start}', exist_ok=True)
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}, slice {start + 5 * i}')
    anim.save(f'./animations/{case}/{start}/{frames}.gif', writer='pillow', progress_callback=progress_callback)


def plot_mean_absolute_speed(umean_abs, x_axis, y_axis, G, wind_vec, layout_file):
    """"
    Plots the mean absolute wind speed over the given grid
    inputs:
    umean_abs = the absolute wind speed data
    x_axis = x value range of the grid
    y_axis = y value range of the grid
    """
    fig, ax = plt.subplots()


    # TODO: make this work with precomputed umean
    plt.imshow(umean_abs, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), origin='lower', aspect='auto')
    plt.colorbar(label='Mean Velocity (UmeanAbs)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interpolated UmeanAbs at Hub-Height')

    # Create wind direction arrow
    pos_dict = nx.get_node_attributes(G, 'pos')
    wind_start = np.mean(np.array(list(pos_dict.values())), axis=0)
    scaled_wind_vec = 1000 * wind_vec

    # Create windmill layout
    df = pd.read_csv(layout_file, sep=",", header=None)

    angle = np.deg2rad(0)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.array([[(x_axis[-1] - x_axis[0] ) / 2, (y_axis[-1] - y_axis[0])/2]])

    # Remove z component
    p = df.values[:, :2]
    rotated_windmills = np.squeeze((R @ (p.T - o.T) + o.T).T)

    for i, (x, y) in enumerate(rotated_windmills):
        circ = Circle((x, y), 100, color='red')
        ax.add_patch(circ)
        ax.text(x, y, f'{i}', ha='center', va='center')

    p = np.array([[scaled_wind_vec[0], scaled_wind_vec[1]]])
    o = np.array([[0, 0]])
    rotated_wind = np.squeeze((R @ (p.T - o.T) + o.T).T)
    print(rotated_wind)
    plt.quiver(wind_start[0], wind_start[1], rotated_wind[0], rotated_wind[1],
               angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')

    plt.show()


def angle_to_vec(wind_angle):
    angle_radians = np.deg2rad(wind_angle)
    return np.array([np.cos(angle_radians), np.sin(angle_radians)])


def read_wind_angles(file):
    return np.genfromtxt(file, delimiter=",")


def get_wind_vec_at_time(wind_angles, timestep):
    return -1 * angle_to_vec(wind_angles[wind_angles[:, 0] < timestep][-1, 1])


def read_turbine_positions(file):
    return np.genfromtxt(file, delimiter=",")[:, :2]


def get_angle_between_vec(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm(v1) * norm(v2)), -1.0, 1.0)))


def create_turbine_graph(pos, wind_vec, max_angle=90, max_dist=np.inf):
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


def plot_graph(G, wind_vec, max_angle=90):
    pos_dict = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 8))
    # Draw nodes
    nx.draw_networkx_nodes(G, pos_dict, node_size=500, node_color='lightblue')
    # Draw edges
    nx.draw_networkx_edges(G, pos_dict, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20)
    # Draw node labels
    nx.draw_networkx_labels(G, pos_dict, font_size=12, font_family='sans-serif')
    # Draw wind direction
    wind_start = np.mean(np.array(list(pos_dict.values())), axis=0)
    scaled_wind_vec = 1000 * wind_vec
    plt.quiver(wind_start[0], wind_start[1], scaled_wind_vec[0], scaled_wind_vec[1],
               angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')
    plt.legend()
    plt.title(f"Turbine Graph with Wind Direction (max angle: {max_angle})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def calculate_wake_distances(turb_vec, wind_vec, angle=None):
    if angle is None:
        angle = get_angle_between_vec(turb_vec, wind_vec)

    angle_rad = np.deg2rad(angle)
    dist_i_j = norm(turb_vec)
    rad_dist = dist_i_j * np.sin(angle_rad)
    down_str_dist = dist_i_j * np.cos(angle_rad)
    return rad_dist, down_str_dist


if __name__ == "__main__":
    case = 1
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"./slices/Case_{case}/HKN_{turbines}_layout_balanced.csv"
    wind_angles = read_wind_angles(f"./slices/Case_{case}/HKN_{turbines}_dir.csv")
    turbine_pos = read_turbine_positions(layout_file)
    timestep = 505
    max_angle = 90
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)
    graph = create_turbine_graph(turbine_pos, wind_vec, max_angle=max_angle)
    plot_graph(graph, wind_vec, max_angle=max_angle)

    # animate_mean_absolute_speed(30005)
    umean_abs, x_axis, y_axis = vtk_to_umean_abs(
        f'./slices/Case_{case}/BL/{30000 + timestep}/U_slice_horizontal.vtk')
    plot_mean_absolute_speed(umean_abs, x_axis, y_axis, graph, wind_vec, layout_file)
    # animate_mean_absolute_speed(30005, comparison=True, case="Case_1")