import numpy as np
from scipy.spatial.transform import Rotation as R


def interpolate_points(p1, p2, num_points):
    # p1 and p2 are tuples representing two points: (x1, y1) and (x2, y2)
    x_values = np.linspace(p1[0], p2[0], num_points)
    y_values = np.linspace(p1[1], p2[1], num_points)

    # Combine the x and y coordinates
    points = np.vstack((x_values, y_values)).T
    return points

def wind_direction_grid(x_turb: list,
                        y_turb: list,
                        x_bounds: tuple,
                        y_bounds: tuple,
                        x_resolution: float,
                        y_resolution: float,
                        wind_dir: list,
                        len_vector: float = 10.):
    """
    :param x_turb: List of x-coordinates for turbines (meters).
    :param y_turb: List of y-coordinates for turbines (meters).
    :param x_bounds: Tuple representing the x-coordinate boundaries (meters).
    :param y_bounds: Tuple representing the y-coordinate boundaries (meters).
    :param x_size: Size of the grid in the x-direction (degrees).
    :param y_size: Size of the grid in the y-direction (degrees).
    :param wind_dir: List representing wind directions.
        If the list has a single element, it represents a global wind direction (degrees).
    :param global_windspeed: Global wind speed. Defaults to 8 (m/s).
    :return: Grid representing wind direction influence over the specified area.
    """
    wind_dir_ = np.array(wind_dir) + 180

    x_scale = x_resolution/(x_bounds[1] - x_bounds[0])
    y_scale = y_resolution/(y_bounds[1] - y_bounds[0])
    grid = np.ones((x_resolution, y_resolution))

    vector = np.array([x_turb,y_turb,[0]*len(x_turb)]).T

    vector[:,0] *= x_scale
    vector[:,1] *= y_scale

    wind_vec = np.copy(vector)
    wind_vec[:, 0] += len_vector

    grid = np.ones((x_resolution, y_resolution))*2

    if len(wind_dir) == 1:
        """ a global wind direction """

        rot = R.from_euler("zxy", angles=[wind_dir_[0], 0, 0], degrees=True)
        new_wind_vec = rot.apply(np.array([10, 0, 0]))
        rot_line = interpolate_points((0, 0), new_wind_vec[:2], 10)

        for i, v in enumerate(vector):
            x_new = rot_line[:, 0] + v[0]
            y_new = rot_line[:, 1] + v[1]

            x_new = x_new.clip(0, x_resolution - 1)
            y_new = y_new.clip(0, y_resolution - 1)

            grid[x_new.astype(int), y_new.astype(int)] = 1
            grid[v[0].astype(int), v[1].astype(int)] = 0

    return grid


