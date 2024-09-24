import os
import numpy as np

from multiprocessing import Pool

from utils import vtk_to_umean_abs

class MultiThread:
    def __init__(self, case, type):
        self.case = case
        self.type = type

    def compute(self, index):
        umean_abs, _, _ = vtk_to_umean_abs(f'./slices/{self.case}/{self.type}/{index}/U_slice_horizontal.vtk')
        np.save(f'./slices/{self.case}/Processed/{self.type}/Windspeed_map_scalars_{index}', umean_abs)
        print(f'Processed {index}')

def preprocess(case, type, overwrite=True):
    dirs = set(os.listdir(f'./slices/{case}/{type}'))
    os.makedirs(f'./slices/{case}/Processed/{type}', exist_ok=True)

    if not overwrite:
        existing = {file.split("_")[-1].split(".")[0] for file in os.listdir(f'./slices/{case}/Processed/{type}')}
        dirs = dirs - existing

    print(f'Processing in total {len(dirs)} files')

    m = MultiThread(case, type)

    with Pool() as p:
        p.map(m.compute, dirs)


if __name__ == '__main__':
    preprocess("Case_1", "BL", overwrite=False)
