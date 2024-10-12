import numpy as np
from matplotlib import pyplot as plt

from utils.extract_windspeed import WindspeedExtractor
from utils.preprocessing import (read_wind_angles, read_turbine_positions, get_wind_vec_at_time,
                                 create_turbine_nx_graph,
                                 read_wind_speed_scalars, resize_windspeed)
from utils.visualization import plot_graph, plot_mean_absolute_speed


def test_graph_creation_plotting():
    case = 1
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"
    layout_file = f"../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    wind_angles = read_wind_angles(f"../data/Case_0{case}/HKN_{turbines}_dir.csv")
    turbine_pos = read_turbine_positions(layout_file)
    timestep = 430
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)

    # Define max_angle values to loop over
    max_angles = [30, 90, 360]

    # Create subplots for each max_angle
    fig, axs = plt.subplots(1, len(max_angles), figsize=(15, 5))

    # Loop through each max_angle and plot the graph on respective axes
    for i, max_angle in enumerate(max_angles):
        graph = create_turbine_nx_graph(turbine_pos, wind_vec, max_angle=max_angle)

        # Pass each subplot axis to plot_graph
        plot_graph(graph, wind_vec, max_angle=max_angle, ax=axs[i])

    # Display all plots side by side
    plt.tight_layout()
    plt.savefig("turbine_graphs.pdf")
    plt.show()

    # animate_mean_absolute_speed(30005)
    # umean_abs, x_axis, y_axis = vtk_to_umean_abs(
    #     f'../data/Case_0{case}/measurements_flow/postProcessing_BL/sliceDataInstantaneous/{30000 + timestep}/U_slice_horizontal.vtk')
    umean_abs = read_wind_speed_scalars("postProcessing_BL", 30000 + timestep, f"Case_0{case}")
    plot_mean_absolute_speed(umean_abs, 100 * wind_vec, layout_file)


def test_resize():
    case = 1
    timestep = 430
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"

    layout_file = f"../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"

    wind_angles = read_wind_angles(f"../data/Case_0{case}/HKN_{turbines}_dir.csv")
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)

    umean_abs = read_wind_speed_scalars("postProcessing_BL", 30000 + timestep, f"Case_0{case}")
    scaled_umean = resize_windspeed(umean_abs, (128, 128))

    plot_mean_absolute_speed(scaled_umean, 100 * wind_vec, layout_file)


def test_extract_windspeed():
    case = 1
    timestep = 430
    turbines = "12_to_15" if case == 1 else "06_to_09" if case == 2 else "00_to_03"

    layout_file = f"../data/Case_0{case}/HKN_{turbines}_layout_balanced.csv"
    turbine_pos = read_turbine_positions(layout_file)

    umean_abs = read_wind_speed_scalars("postProcessing_BL", 30000 + timestep, f"Case_0{case}")
    scaled_umean = resize_windspeed(umean_abs, (150, 150))

    wind_angles = read_wind_angles(f"../data/Case_0{case}/HKN_{turbines}_dir.csv")
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)
    wind_angle = wind_angles[wind_angles[:, 0] < timestep][-1, 1]
    yaw_angles = np.arange(start=-45, stop=45, step=9)

    blade_pixels = []

    extractor = WindspeedExtractor(turbine_pos, umean_abs)

    extractor(umean_abs, wind_angle, yaw_angles)
    extractor(umean_abs, wind_angle, yaw_angles, blade_pixels)

    plot_mean_absolute_speed(scaled_umean, 100 * wind_vec, layout_file, blade_pixels)

if __name__ == "__main__":
    test_graph_creation_plotting()
