import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from utils.preprocessing import read_wind_speed_scalars


def animate_mean_absolute_speed(start, frames=None, comparison=False, case="Case_01"):
    """
    Creates an animation of one or two wind speed maps over time, in 5 second increments.
    Uses the preprocessed data located in ./slices/Processed/BL/*_xxxxx.npy files, containing the
    precomputed mean absolute wind speed.

    Args:
        start (int): Start timestamp to render from.
        frames (Optional[int]): Amount of frames to render, leave empty to render till the end.
        comparison (bool): Change to True if you want to create two plots, one with wake steering and one without.
        case (str): The case name to use for file paths.
    """
    if frames is None:
        dirs = os.listdir(f'../data/{case}/measurements_flow')
        all_files = [os.listdir(f'../data/{case}/measurements_flow/{dir}/windspeedMapScalars') for dir in dirs]
        biggest_file = [int(max(filter(lambda file: file != "README.md", files),
                                key=lambda x: int(x.split('.')[0].split('_')[-1])).split('.')[0].split('_')[-1]) for
                        files in all_files]
        frames = (min(biggest_file) - start) // 5 + 1
        print(frames)

    images = []

    def setup_image(ax, type):
        umean = read_wind_speed_scalars(type, start, case)
        axes_image = ax.imshow(umean, animated=True, vmin=0, vmax=10)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Distance (m)")
        images.append((axes_image, type))

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
        for axes_image, type in images:
            umean_abs = read_wind_speed_scalars(type, start + 5 * i, case)
            axes_image.set_data(umean_abs)
        return fig, *images

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=frames, interval=50)
    os.makedirs(f'./animations/{case}/{start}', exist_ok=True)
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}, slice {start + 5 * i}')
    anim.save(f'./animations/{case}/{start}/{frames}.gif', writer='pillow', progress_callback=progress_callback)


def add_windmills(ax, layout_file, image_size=128):
    """
    Adds windmill layout to the provided axis.

    Args:
        ax (plt.Axes): The axis to which the windmills will be added.
        layout_file (str): CSV file path containing the windmill layout.
        image_size (int): Size of the image for windmills.
    """
    df = pd.read_csv(layout_file, sep=",", header=None)
    scale_factor = image_size / 5000

    for i, (x, y, z) in enumerate(df.values * scale_factor):
        circ = Circle((x, y), image_size / 60, color='red')
        ax.add_patch(circ)
        ax.text(x, y, f'{i}', ha='center', va='center')


def add_blades(ax, windmill_blades):
    """
   Adds blades to the windmills on the provided axis.

   Args:
       ax (plt.Axes): The axis to which the blades will be added.
       windmill_blades (List[np.ndarray]): List of arrays representing the positions of windmill blades.
   """
    for blade in windmill_blades:
        start = blade[0]
        end = blade[-1]
        ax.add_line(Line2D([start[0], end[0]], [start[1], end[1]], color='red', lw=3))


def add_quiver(ax, wind_vec, center):
    """
    Adds a quiver (arrow) representing wind direction to the provided axis.

    Args:
        ax (plt.Axes): The axis to which the quiver will be added.
        wind_vec (np.ndarray): The wind vector.
        center (float): The center point for the quiver arrow.
    """
    ax.quiver(center, center, wind_vec[0], wind_vec[1],
              angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')


def add_imshow(fig, ax, umean_abs, color_bar=True):
    """
    Displays the mean absolute wind speed on the given axis.

    Args:
        fig (plt.Figure): The figure to which the image will be added.
        ax (plt.Axes): The axis to plot on.
        umean_abs (np.ndarray): The absolute wind speed data.
        color_bar (bool): Whether to include a color bar.

    Returns:
        plt.imshow: The image object created.
    """
    axesImage = ax.imshow(umean_abs, extent=(0, 128, 0, 128), origin='lower', aspect='equal', vmin=0, vmax=10)
    if color_bar:
        fig.colorbar(axesImage, ax=ax, label='Mean Velocity (UmeanAbs)')
    return axesImage


def get_mean_absolute_speed_figure(umean_abs, wind_vec, layout_file=None, windmill_blades=None):
    """
    Creates a figure displaying the mean absolute wind speed, wind direction, and windmill layout.

    Args:
        umean_abs (np.ndarray): The absolute wind speed data.
        wind_vec (np.ndarray): The wind vector data.
        layout_file (Optional[str]): File for windmill layout (optional).
        windmill_blades (Optional[List[np.ndarray]]): Blade configuration for windmills (optional).

    Returns:
        plt.Figure: The created figure.
    """
    fig, ax = plt.subplots()

    add_imshow(fig, ax, umean_abs)
    add_quiver(ax, wind_vec / 2, umean_abs.shape[0] / 2)
    if windmill_blades:
        add_blades(ax, windmill_blades)
    else:
        add_windmills(ax, layout_file, umean_abs.shape[0])
    return fig


def plot_mean_absolute_speed(umean_abs, wind_vec, layout_file=None, windmill_blades=None):
    """
    Plots the mean absolute wind speed over the given grid.

    Args:
        umean_abs (np.ndarray): The absolute wind speed data.
        wind_vec (np.ndarray): The wind vector data.
        layout_file (Optional[str]): File for windmill layout (optional).
        windmill_blades (Optional[List[np.ndarray]]): Blade configuration for windmills (optional).
    """
    fig, ax = plt.subplots()
    plot_mean_absolute_speed_subplot(ax, umean_abs, wind_vec, layout_file=layout_file, windmill_blades=windmill_blades)
    plt.show()


def plot_mean_absolute_speed_subplot(ax, umean_abs, wind_vec, layout_file=None, windmill_blades=None, color_bar=True):
    """
    Plots the mean absolute wind speed on a given axis.

    Args:
        ax (plt.Axes): Axis to plot on.
        umean_abs (np.ndarray): The absolute wind speed data.
        wind_vec (np.ndarray): The wind vector data.
        layout_file (Optional[str]): File for windmill layout (optional).
        windmill_blades (Optional[List[np.ndarray]]): Blade configuration for windmills (optional).
        color_bar (bool): Whether to include a color bar.

    Returns:
        plt.imshow: The image object created.
    """
    img = add_imshow(ax.figure, ax, umean_abs, color_bar=color_bar)
    add_quiver(ax, wind_vec / 2, umean_abs.shape[0] / 2)
    if windmill_blades:
        add_blades(ax, windmill_blades)
    else:
        add_windmills(ax, layout_file, umean_abs.shape[0])
    return img


def plot_graph(G, wind_vec, max_angle=90, ax=None):
    """
    Plots a directed graph with nodes and edges representing the wind turbines.

    Args:
        G (nx.Graph): The graph to plot.
        wind_vec (np.ndarray): The wind vector data.
        max_angle (float): Maximum angle for the arrows in the plot.
        ax (Optional[plt.Axes]): The axis to plot on, if None a new axis will be created.
    """
    pos_dict = nx.get_node_attributes(G, 'pos')

    # Use the axis if provided, otherwise use the current figure
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    # Draw nodes
    nx.draw_networkx_nodes(G, pos_dict, node_size=500, node_color='lightblue', ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, pos_dict, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, ax=ax)
    # Draw node labels
    nx.draw_networkx_labels(G, pos_dict, font_size=12, font_family='sans-serif', ax=ax)

    # Draw wind direction
    wind_start = np.mean(np.array(list(pos_dict.values())), axis=0)
    scaled_wind_vec = 500 * wind_vec
    ax.quiver(wind_start[0], wind_start[1], scaled_wind_vec[0], scaled_wind_vec[1],
              angles='xy', scale_units='xy', scale=1, color='red', label='Wind Direction')

    ax.legend()
    ax.set_title(f"Max Angle: {max_angle}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.set_aspect('equal')


def plot_prediction_vs_real(predicted, target, case=1, number=0):
    """
    Plots the predicted wind speed versus the target wind speed.

    Args:
        predicted (np.ndarray): The predicted wind speed data.
        target (np.ndarray): The target wind speed data.
        case (int, optional): Case number to determine windmill layout file. Defaults to 1.
        number (int, optional): Number for saving the plot file. Defaults to 0.
    """
    layout_file = get_layout_file(case)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot target
    img1 = add_imshow(fig, axs[0], target, color_bar=False)
    axs[0].set_title('Target')
    axs[0].set_aspect('equal', adjustable='box')  # Maintain aspect ratio
    add_windmills(axs[0], layout_file)

    # Plot predicted
    add_imshow(fig, axs[1], predicted, color_bar=False)
    axs[1].set_title('Predicted')
    axs[1].set_aspect('equal', adjustable='box')  # Maintain aspect ratio
    add_windmills(axs[1], layout_file)

    cbar_ax = fig.add_axes([1.02, 0, 0.02, 1])  # [left, bottom, width, height]
    fig.colorbar(img1, cax=cbar_ax, orientation='vertical')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'predictions_vs_targets_{number}.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def animate_prediction_vs_real(umean_callback, n_frames=100, file_path="animation"):
    """
    Creates an animation comparing the predicted and target wind speed over frames.

    Args:
        umean_callback (Callable[[int], Tuple[np.ndarray, np.ndarray]]): Function that returns target and predicted wind speeds given a frame index.
        n_frames (int, optional): Number of frames for the animation. Defaults to 100.
        file_path (str, optional): Path to save the animation file. Defaults to "animation".
    """
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
    """
    Retrieves the layout file path for wind turbines based on the case number.

    Args:
        case (int): The case number to determine the layout file.

    Returns:
        str: The file path of the layout CSV.
    """
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    return f"../../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
