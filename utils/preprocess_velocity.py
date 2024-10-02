import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import griddata

from tqdm import tqdm

from preprocessing import import_vtk


base_path = "postProcessing_LuT2deg_internal/postProcessing_LuT2deg_internal/sliceDataInstantaneous"
all_time_instances = os.listdir(base_path)

def get_velocity_from_vtk(time_step: str):
    _, centers, data = import_vtk(f"{base_path}/{time_step}/U_slice_horizontal.vtk")

    x_axis = np.linspace(np.min(centers[:, 0]), np.max(centers[:, 0]), 300)
    y_axis = np.linspace(np.min(centers[:, 1]), np.max(centers[:, 1]), 300)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    u = griddata(centers[:, 0:2], data[:, 0], (x_grid, y_grid), method='linear')
    v = griddata(centers[:, 0:2], data[:, 1], (x_grid, y_grid), method='linear')
    w = griddata(centers[:, 0:2], data[:, 2], (x_grid, y_grid), method='linear')
    field = np.stack([u, v, w])

    np.save(f"data/Case_01/measurements_flow/postProcessing_LuT2deg_internal/winSpeedMapVector/{time_step}.npy", field)

def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        for _ in tqdm(executor.map(get_velocity_from_vtk, all_time_instances), total=len(all_time_instances)):
            pass


if __name__ == "__main__":
    main()