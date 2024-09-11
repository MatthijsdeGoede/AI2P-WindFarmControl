import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os

import matplotlib.animation as animation


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


def animate_mean_absolute_speed(start, frames=None):
    if frames is None:
        all_files = os.listdir('./slices/BL/')
        biggest_file = int(max(all_files, key=lambda x: int(x.split('.')[0])))
        frames = (biggest_file - start)//5
        print(frames)
    umean_abs, x_axis, y_axis = vtk_to_umean_abs(
        f'./slices/BL/{start}/U_slice_horizontal.vtk')

    fig, ax = plt.subplots()
    axesImage = ax.imshow(umean_abs, animated=True)
    fig.colorbar(axesImage, ax=ax)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(f"Interpolated UmeanAbs at Hub-Height")
    time_stamp = ax.text(0.5, 0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    def animate(i):
        umean_abs, x_axis, y_axis = vtk_to_umean_abs(
            f'./slices/BL/{start + 5 * i}/U_slice_horizontal.vtk')
        axesImage.set_data(umean_abs)
        time_stamp.set_text(f't = {5 * i} minutes')
        print(f'Completed slice #{start + 5 * i}')
        return axesImage, time_stamp

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=frames, interval=100, blit=True)
    os.makedirs(f'./animations/{start}', exist_ok=True)
    anim.save(f'./animations/{start}/{frames}.gif', writer='imagemagick')


def plot_mean_absolute_speed(umean_abs, x_axis, y_axis):
    """"
    Plots the mean absolute wind speed over the given grid
    inputs:
    umean_abs = the absolute wind speed data
    x_axis = x value range of the grid
    y_axis = y value range of the grid
    """
    plt.imshow(umean_abs, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), origin='lower', aspect='auto')
    plt.colorbar(label='Mean Velocity (UmeanAbs)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interpolated UmeanAbs at Hub-Height')
    plt.show()


if __name__ == "__main__":
    animate_mean_absolute_speed(30005, frames=500)
    # umean_abs, x_axis, y_axis = vtk_to_umean_abs(
    #     '../measurements_flow/postProcessing_BL/sliceDataInstantaneous/30890/U_slice_horizontal.vtk')
    # plot_mean_absolute_speed(umean_abs, x_axis, y_axis)
