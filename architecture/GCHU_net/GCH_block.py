from floris import FlorisModel
import numpy as np
from scipy.interpolate import griddata
import torch.nn as nn
from torch import float32
from torch import from_numpy

class GCH(nn.Module):
    """
    A class representing the GCH (Generic Computational Hydrodynamics) model.

    The class `GCH` is designed to interact with the FlorisModel,
    allowing for computations on a specified grid and extraction of processed data.

    """
    def __init__(self,
                 x_resolution: int,
                 y_resolution: int,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 x_size: float,
                 y_size: float,
                 height: float,
                 ):
        super().__init__()

        """
        :param x_resolution: Resolution in the x-axis.
        :param y_resolution: Resolution in the y-axis.
        :param x_bounds: Bounds for the x-axis as a tuple (min, max) in meters.
        :param y_bounds: Bounds for the y-axis as a tuple (min, max) in meters.
        :param x_size: Size along the x-axis in pixels.
        :param y_size: Size along the y-axis in pixels.
        :param height: Height value in meters.
        """

        self.fmodel = FlorisModel("GCHU_net/gch.yaml")
        # Placeholder for future implementations list[float], yaw_angles: list[float]):

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.x_size = x_size
        self.y_size = y_size
        self.height = height


    def forward(self,
                x_coordinates_turbines: list[float],
                y_coordinates_turbines: list[float],
                wind_directions: list[float],
                wind_speeds: list[float],
                yaw_angles: list[float],
                ):
        """
        :param x_coordinates_turbines: List of x-coordinates for each turbine in meters.
        :param y_coordinates_turbines: List of y-coordinates for each turbine in meters.
        :param wind_directions: List of wind directions in degrees.
        :param wind_speeds: List of wind speeds in m/s.
        :param yaw_angles: List of yaw angles for each turbine in degrees.
        :return: 4-dimensional numpy array representing the mean absolute wind speed on the grid.
        """
        self.fmodel.set(layout_x=x_coordinates_turbines,
                    layout_y=y_coordinates_turbines,
                    wind_directions=wind_directions,
                    wind_speeds=wind_speeds,
                    yaw_angles=np.array(yaw_angles).reshape((1, len(yaw_angles)))
                    )

        x_max = np.max(self.x_bounds)
        y_max = np.max(self.y_bounds)
        x_min = np.min(self.x_bounds)
        y_min = np.min(self.y_bounds)

        x_grid_len = x_max - x_min
        y_grid_len = x_max - x_min

        bound_margin_rot = 0
        if x_grid_len == y_grid_len:

            # if the grid is a rectangle this avoids creating gaps in the coreners of the grid
            bound_margin_rot = x_grid_len * np.sqrt(2 / np.sqrt(2) - 1.25) / 2

        horizontal_plane = self.fmodel.calculate_horizontal_plane(height=self.height,
                                                                  x_resolution=self.x_resolution,
                                                                  y_resolution=self.y_resolution,
                                                                  x_bounds=(-bound_margin_rot+x_min,
                                                                            x_max+bound_margin_rot),
                                                                  y_bounds=(-bound_margin_rot+y_min,
                                                                           y_max+bound_margin_rot))

        x_grid = horizontal_plane.df["x1"].to_numpy()
        y_grid = horizontal_plane.df["x2"].to_numpy()

        u = horizontal_plane.df["u"].to_numpy()

        Xaxis = np.linspace(x_min, x_max, self.x_size)
        Yaxis = np.linspace(y_min ,y_max, self.y_size)

        Xm, Ym = np.meshgrid(Xaxis, Yaxis)

        coordinates = np.column_stack((x_grid, y_grid))

        UmeanAbs = griddata(coordinates, u, (Xm, Ym), method='linear')

        # Check for NaN values and replace them (e.g., with zeros or some reasonable default value)
        UmeanAbs = np.nan_to_num(UmeanAbs, nan=0.0)

        UmeanAbs = UmeanAbs.reshape((1, 1, self.x_size, self.y_size))

        UmeanAbs = from_numpy(UmeanAbs)

        return UmeanAbs.to(float32)
