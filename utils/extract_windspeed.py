import numpy as np

ROTOR_DIAMETER = 178.3

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def rotate_and_translate(points, wind_vec, yaw_angle):
    rotor_diameter = len(points)
    translated = np.arange(start=-rotor_diameter // 2 + 1, stop=rotor_diameter // 2 + 1, step=1)
    translated = np.stack((np.zeros(rotor_diameter), translated), axis=1)
    translated = points + translated
    return rotate(translated, origin=points[0], degrees=wind_vec + yaw_angle)

# def get_turbine_rotor_pixels(turbine_location, wind_vec, yaw_angle):


def extract_wind_speed(windspeedmap, turbine_locations, wind_vec, yaw_angles, location_pixels=None):
    n_turbines = len(turbine_locations)

    turbine_locations = np.array(turbine_locations)
    scale_factor = windspeedmap.shape[0] / 5000
    print(f"One pixel = {round(1/scale_factor, 2)}m")
    print(turbine_locations * scale_factor)

    rotor_diameter_pixels = int(round(ROTOR_DIAMETER * scale_factor, 0))
    print(f"Rotor diameter = {rotor_diameter_pixels}pixels")

    turbine_locations = np.repeat(np.round(turbine_locations * scale_factor, 0), rotor_diameter_pixels, axis=0).reshape((n_turbines, rotor_diameter_pixels, -1))

    wind_speed_at_turbine = np.empty((n_turbines, rotor_diameter_pixels))

    for i, turbine_location in enumerate(turbine_locations):
        rotated = rotate_and_translate(turbine_location, wind_vec, yaw_angles[i]).astype(int)
        print("Pixels of interest: ", rotated)

        if location_pixels is not None:
            location_pixels.append(rotated)

        results = np.array([windspeedmap[index[0], index[1]] for index in rotated])
        print("Results: ", results)

        wind_speed_at_turbine[i] = results

    means = np.mean(wind_speed_at_turbine, axis=1)
    print("Means: ", means)

    return means