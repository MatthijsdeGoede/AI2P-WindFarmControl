import numpy as np
import pandas as pd

ROTOR_DIAMETER = 178.3


class WindSpeedExtractor:
    def __init__(self, turbine_locations, map_size):
        self.n_turbines = len(turbine_locations)

        scale_factor = map_size / 5000
        # print(f"One pixel = {round(1/scale_factor, 2)}m")

        self.turbine_location_centers = np.array(turbine_locations) * scale_factor

        self.rotor_diameter_pixels = int(round(ROTOR_DIAMETER * scale_factor, 0))
        # print(f"Rotor diameter = {self.rotor_diameter_pixels}pixels")

        turbine_locations = np.repeat(np.round(self.turbine_location_centers, 0), self.rotor_diameter_pixels,
                                      axis=0).reshape((self.n_turbines, self.rotor_diameter_pixels, -1))

        translate = np.arange(start=-self.rotor_diameter_pixels // 2 + 1, stop=self.rotor_diameter_pixels // 2 + 1,
                              step=1)
        translate = np.stack((np.ones(self.rotor_diameter_pixels) * -2, translate), axis=1)

        self.turbine_locations = turbine_locations + translate

    def rotate(self, p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    def __call__(self, wind_speed_map, wind_angle, yaw_angles, location_pixels=None):

        wind_speeds_at_turbine = np.empty((self.n_turbines, self.rotor_diameter_pixels))

        for i, turbine_location in enumerate(self.turbine_locations):
            rotated = self.rotate(turbine_location, origin=self.turbine_location_centers[i],
                                  degrees=yaw_angles[i]).astype(int)

            if location_pixels is not None:
                location_pixels.append(rotated)

            wind_speeds_at_turbine[i] = np.array([wind_speed_map[index[1], index[0]] for index in rotated])

        means = np.mean(wind_speeds_at_turbine, axis=1)

        return means
