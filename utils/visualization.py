import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Circle

from utils.preprocessing import read_wind_angles, read_turbine_positions, get_wind_vec_at_time, create_turbine_nx_graph, \
    vtk_to_umean_abs


def get_data_from_file(type, i, case):
    return np.load(f'../data/{case}/measurements_flow/{type}/windspeedMapScalars/Windspeed_map_scalars_{i}.npy')


def animate_mean_absolute_speed(start, frames=None, comparison=False, case="Case_01"):
    """
    Creates an animation of one or two wind speed maps over time, in 5 second increments. Uses the preprocessed data.
    Data is expected to be located in ./slices/Processed/BL/*_xxxxx.npy files, and contain the precomputed mean
    absolute wind speed. xxxxx denotes the timestamp of the wind speed measurement in seconds.
    @param start: Start timestamp to render from.
    @param frames: Amount of frames to render, leave empty to render till the end.
    @param comparison: Change to True if you want to create two plots, one with wake steering and one without.
    """
    if frames is None:
        dirs = os.listdir(f'../data/{case}/measurements_flow')
        all_files = [os.listdir(f'../data/{case}/measurements_flow/{dir}/windspeedMapScalars') for dir in dirs]
        biggest_file = [int(max(filter(lambda file: file != "README.md", files), key=lambda x: int(x.split('.')[0].split('_')[-1])).split('.')[0].split('_')[-1]) for files in all_files]
        frames = (min(biggest_file) - start)//5 + 1
        print(frames)

    images = []

    def setup_image(ax, type):
        umean = get_data_from_file(type, start, case)
        axes_image = ax.imshow(umean, animated=True)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Distance (m)")
        images.append((axes_image, type))

    # fig, (ax1) = plt.subplots(1, 1)
    #
    # setup_image(ax1, "BL")

    # setup_image(ax2, "LuT2deg")
    # fig.colorbar(axes_image_1)
    if comparison:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        setup_image(ax1, "postProcessing_BL")
        ax1.set_title("Greedy Controller")
        setup_image(ax2, "postProcessing_LuT2deg_internal")
        ax2.set_title("Wake Steering")
    else:
        fig, (ax1) = plt.subplots(1, 1)
        setup_image(ax1, "postProcessing_BL")
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


def add_windmills(ax, layout_file):
    # Create windmill layout
    df = pd.read_csv(layout_file, sep=",", header=None)

    for i, (x, y, z) in enumerate(df.values):
        circ = Circle((x, y), 100, color='red')
        ax.add_patch(circ)
        ax.text(x, y, f'{i}', ha='center', va='center')


def add_quiver(ax, x_axis, y_axis, wind_vec):
    scaled_wind_vec = 1000 * wind_vec

    ax.quiver((x_axis[-1] - x_axis[0])/2, (y_axis[-1] - y_axis[0])/2, scaled_wind_vec[0], scaled_wind_vec[1],
               angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')


def plot_mean_absolute_speed(umean_abs, x_axis, y_axis, wind_vec, layout_file):
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

    add_windmills(ax, layout_file)
    add_quiver(ax, x_axis, y_axis, wind_vec)

    plt.show()


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


def plot_prediction_vs_real(predicted, target):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot predicted
    axs[0].imshow(target, extent=(0, 300, 0, 300), origin='lower', aspect='auto')
    axs[0].set_title('Target UmeanAbs')
    axs[0].set_xlabel('X-axis')
    axs[0].set_ylabel('Y-axis')
    cbar1 = plt.colorbar(axs[0].imshow(target), ax=axs[0])
    cbar1.set_label('Mean Velocity (UmeanAbs)')

    # Plot target
    axs[1].imshow(predicted, extent=(0, 300, 0, 300), origin='lower', aspect='auto')
    axs[1].set_title('Predicted UmeanAbs')
    axs[1].set_xlabel('X-axis')
    axs[1].set_ylabel('Y-axis')
    cbar2 = plt.colorbar(axs[1].imshow(predicted), ax=axs[1])
    cbar2.set_label('Mean Velocity (UmeanPredicted)')

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    case = 1
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    wind_angles = read_wind_angles(f"../data/Case_0{case}/HKN_{turbines}_dir.csv")
    turbine_pos = read_turbine_positions(layout_file)
    timestep = 430
    max_angle = 90
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)

    # Plot graph of windmills
    graph = create_turbine_nx_graph(turbine_pos, wind_vec, max_angle=max_angle)
    plot_graph(graph, wind_vec, max_angle=max_angle)

    # animate_mean_absolute_speed(30005)
    umean_abs, x_axis, y_axis = vtk_to_umean_abs(
        f'../data/Case_0{case}/measurements_flow/postProcessing_BL/sliceDataInstantaneous/{30000 + timestep}/U_slice_horizontal.vtk')
    plot_mean_absolute_speed(umean_abs, x_axis, y_axis, wind_vec, layout_file)
    # animate_mean_absolute_speed(30005, comparison=True, case="Case_1")