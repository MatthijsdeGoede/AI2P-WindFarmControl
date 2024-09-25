import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import griddata

from tqdm import tqdm

from utils import import_vtk

base_path = "data/raw/case1"
total_files = os.listdir(base_path)

def get_velocity_from_vtk(file_path: str):
    _, centers, data = import_vtk(f"{base_path}/{file_path}")

    x_axis = np.linspace(np.min(centers[:, 0]), np.max(centers[:, 0]), 300)
    y_axis = np.linspace(np.min(centers[:, 1]), np.max(centers[:, 1]), 300)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    u = griddata(centers[:, 0:2], data[:, 0], (x_grid, y_grid), method='linear')
    v = griddata(centers[:, 0:2], data[:, 1], (x_grid, y_grid), method='linear')
    w = griddata(centers[:, 0:2], data[:, 2], (x_grid, y_grid), method='linear')
    field = np.stack([u, v, w])

    timestep = file_path.split(".")[0]
    np.save(f"data/preprocessed/case1/{timestep}.npy", field)

def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        for _ in tqdm(executor.map(get_velocity_from_vtk, total_files), total=len(total_files)):
            pass


if __name__ == "__main__":
    main()