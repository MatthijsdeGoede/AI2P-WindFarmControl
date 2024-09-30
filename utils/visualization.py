import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Circle

from utils.preprocessing import read_wind_speed_scalars


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
        umean = read_wind_speed_scalars(type, start, case)
        axes_image = ax.imshow(umean, animated=True)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Distance (m)")
        images.append((axes_image, type))

    # fig, (ax1) = plt.subplots(1, 1)
    #
    # setup_image(ax1, "BL")

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
            umean_abs = read_wind_speed_scalars(type, start + 5 * i, case)
            axes_image.set_data(umean_abs)
        return fig, *images

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=frames, interval=50)
    os.makedirs(f'./animations/{case}/{start}', exist_ok=True)
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}, slice {start + 5 * i}')
    anim.save(f'./animations/{case}/{start}/{frames}.gif', writer='pillow', progress_callback=progress_callback)


def add_windmills(ax, layout_file):
    # Create windmill layout
    df = pd.read_csv(layout_file, sep=",", header=None)
    scale_factor = 300/5000

    for i, (x, y, z) in enumerate(df.values * scale_factor):
        circ = Circle((x, y), 5, color='red')
        ax.add_patch(circ)
        ax.text(x, y, f'{i}', ha='center', va='center')


def add_quiver(ax, wind_vec):
    ax.quiver(150, 150, wind_vec[0], wind_vec[1],
               angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')

def add_imshow(fig, ax, umean_abs):
    axesImage = ax.imshow(umean_abs, extent=(0, 300, 0, 300), origin='lower', aspect='auto')
    fig.colorbar(axesImage, ax=ax, label='Mean Velocity (UmeanAbs)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    return axesImage

def plot_mean_absolute_speed(umean_abs, wind_vec, layout_file):
    """"
    Plots the mean absolute wind speed over the given grid
    inputs:
    umean_abs = the absolute wind speed data
    x_axis = x value range of the grid
    y_axis = y value range of the grid
    """
    fig, ax = plt.subplots()

    add_imshow(fig, ax, umean_abs)
    add_windmills(ax, layout_file)
    add_quiver(ax, wind_vec)

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


def plot_prediction_vs_real(predicted, target, case=1):

    layout_file = get_layout_file(case)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot target
    add_imshow(fig, axs[0], target)
    axs[0].set_title('Target UmeanAbs')
    add_windmills(axs[0], layout_file)

    # Plot predicted
    add_imshow(fig, axs[1], predicted)
    axs[1].set_title('Predicted UmeanAbs')
    add_windmills(axs[1], layout_file)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def animate_prediction_vs_real(umean_callback, n_frames=100, file_path="animation"):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    target, prediction = umean_callback(0)
    axis_image_target = add_imshow(fig, axs[0], target)
    axis_image_predicted = add_imshow(fig, axs[1], prediction)

    def animate(i):
        target, prediction = umean_callback(i)
        axis_image_target.set_data(target)
        axis_image_predicted.set_data(prediction)

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=n_frames, interval=50)
    os.makedirs(f'{file_path}', exist_ok=True)
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
    anim.save(f'{file_path}/{n_frames}.gif', writer='pillow', progress_callback=progress_callback)


def get_layout_file(case):
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    return f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"