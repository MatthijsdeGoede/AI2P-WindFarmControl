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
    max_angle = 90
    wind_vec = get_wind_vec_at_time(wind_angles, timestep)
    graph = create_turbine_nx_graph(turbine_pos, wind_vec, max_angle=max_angle)
    plot_graph(graph, wind_vec, max_angle=max_angle)

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


if __name__ == "__main__":
    test_graph_creation_plotting()
