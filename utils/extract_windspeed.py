import numpy as np

ROTOR_DIAMETER = 178.3  # Rotor diameter in some unit (e.g., meters)


class WindSpeedExtractor:
    """
    A class to extract wind speeds at specified turbine locations based on a wind speed map.

    Attributes:
        n_turbines (int): Number of wind turbines.
        offset (np.ndarray): Offset for translating turbine positions.
        turbine_location_centers (np.ndarray): Adjusted turbine locations scaled to the map size.
        rotor_diameter_pixels (int): Rotor diameter represented in pixels.
        turbine_locations (np.ndarray): 2D array of turbine locations adjusted for rotor diameter.
        weights (np.ndarray): Weights used for averaging wind speeds based on rotor geometry.
    """

    def __init__(self, turbine_locations, map_size):
        """
        Initializes the WindSpeedExtractor with turbine locations and the map size.

        Args:
            turbine_locations (list of tuples): A list of (x, y) coordinates for each turbine.
            map_size (float): Size of the map, used for scaling turbine locations.
        """
        self.n_turbines = len(turbine_locations)

        # Calculate scale factor based on the given map size
        scale_factor = map_size / 5000

        # Define the offset for turbine positions
        self.offset = np.array([-4, 0])

        # Scale the turbine locations
        self.turbine_location_centers = (np.array(turbine_locations) * scale_factor)

        # Calculate rotor diameter in pixels
        self.rotor_diameter_pixels = int(round(ROTOR_DIAMETER * scale_factor, 0)) + 2

        # Prepare turbine locations by replicating for rotor diameter
        turbine_locations = np.repeat(np.round((np.array(turbine_locations) * scale_factor), 0),
                                      self.rotor_diameter_pixels, axis=0).reshape(
            (self.n_turbines, self.rotor_diameter_pixels, -1))

        # Create translation offsets for rotor positioning
        translate = np.arange(start=-self.rotor_diameter_pixels // 2 + 1,
                              stop=self.rotor_diameter_pixels // 2 + 1, step=1)
        translate = np.stack((np.zeros(self.rotor_diameter_pixels), translate), axis=1)

        # Adjust turbine locations by the translation offsets
        self.turbine_locations = turbine_locations + translate

        # Create weights based on rotor geometry
        x = np.linspace(0, np.pi, self.rotor_diameter_pixels)
        self.weights = np.sin(x)

    def rotate(self, p, origin=(0, 0), degrees=0):
        """
        Rotates a point or set of points around a specified origin by a given angle.

        Args:
            p (np.ndarray): Array of points to rotate, shape (N, 2).
            origin (tuple, optional): The pivot point for rotation (default: (0, 0)).
            degrees (float): The angle in degrees to rotate the points.

        Returns:
            np.ndarray: The rotated points, same shape as input.
        """
        assert not isinstance(degrees, np.ndarray), f"degrees should not be a ndarray, got {type(degrees)}"
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    def __call__(self, wind_speed_map, wind_angle, yaw_angles, location_pixels=None):
        """
        Extracts wind speeds at the turbine locations based on the wind speed map.

        Args:
            wind_speed_map (np.ndarray): 2D array representing wind speeds at each pixel in the map.
            wind_angle (float): The angle of the wind in degrees.
            yaw_angles (list of floats): Yaw angles for each turbine.
            location_pixels (list, optional): List to store turbine pixel locations (default: None).

        Returns:
            np.ndarray: Array of average wind speeds at each turbine location.
        """
        wind_speeds_at_turbine = np.empty((self.n_turbines, self.rotor_diameter_pixels))

        # Rotate the offset based on the wind angle
        translation = self.rotate(self.offset, degrees=wind_angle)

        for i, turbine_location in enumerate(self.turbine_locations):
            # Rotate turbine locations according to the wind angle
            rotated = self.rotate(turbine_location, origin=self.turbine_location_centers[i],
                                  degrees=wind_angle)
            rotated = (rotated + translation).astype(int)  # Translate and convert to integer indices

            # Optionally compute and store the pixel locations of the turbines
            if location_pixels is not None:
                location_pix = self.rotate(turbine_location, origin=self.turbine_location_centers[i],
                                           degrees=yaw_angles[i]).astype(int)
                location_pix = (location_pix + translation).astype(int)
                location_pixels.append(location_pix)

            # Extract wind speeds from the wind speed map at the turbine locations
            wind_speeds_at_turbine[i] = np.array([wind_speed_map[index[1], index[0]] for index in rotated])

        # Calculate the average wind speed at each turbine, weighted by rotor geometry
        means = np.average(wind_speeds_at_turbine, axis=1, weights=self.weights)

        return means
