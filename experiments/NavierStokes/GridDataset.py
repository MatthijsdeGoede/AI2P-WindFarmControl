import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class GridDataset(Dataset):
    def __init__(self, dir, turbine_csv, wind_csv, only_grid_values=False):
        super().__init__()
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.files.remove("README.md")

        self.only_grid_values = only_grid_values
        
        flat_index = torch.arange(90000)
        x_coords = flat_index // 300
        y_coords = flat_index % 300
        self.coords = torch.stack([x_coords, y_coords]).T

        self.df_turbines = pd.read_csv(turbine_csv, index_col=0)
        self.df_wind = pd.read_csv(wind_csv, index_col=0)

    def get_turbine_data(self, time):
        time_instance = self.df_turbines[self.df_turbines['time'] == time][['speed', 'yaw_sin', "yaw_cos"]]
        return torch.from_numpy(time_instance.to_numpy().flatten()).unsqueeze(0)
    

    def get_wind_data(self, time):
        time_instance = self.df_wind[self.df_wind['time'] == time][['winddir_sin', 'winddir_cos']]
        return torch.from_numpy(time_instance.to_numpy())


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if (isinstance(index, int) or isinstance(index, np.int32)):
            #3x300x300
            flow_field = torch.from_numpy(np.load(f"{self.dir}{self.files[index]}"))
            flow_field = flow_field.permute(1, 2, 0).flatten(0, 1) #90000x3

            time = int(self.files[index].split(".")[0])
            time_tensor = torch.full((self.coords.shape[0], 1), time)

            if (self.only_grid_values):
                inputs = torch.concat((self.coords, time_tensor), dim=1)
                return inputs, flow_field

            turbine_data = self.get_turbine_data(time)
            turbine_data = turbine_data.repeat(time_tensor.shape[0], 1)

            wind_data = self.get_wind_data(time)
            wind_data = wind_data.repeat(time_tensor.shape[0], 1)

            inputs = torch.concat((self.coords, time_tensor, turbine_data, wind_data), dim=1)

            return inputs, flow_field

        # elif (isinstance(index, slice)):
        #     time = list(map(lambda x: int(x.split(".")[0]), self.files[index]))
        #     flow_field = np.stack([np.load(f"{self.dir}{file}") for file in self.files[index]])

        # elif (torch.is_tensor(index)):
        #     index = index.tolist()
        #     time = list(map(lambda x: int(x.split(".")[0]), self.files[index]))
        #     flow_field = np.stack([np.load(f"{self.dir}{file}") for file in self.files[index]])            

        else:
            raise TypeError(f"{type(index)}")
    

class RandomGridDataset(Dataset):
    def __init__(self, dir, samples_per_grid=64, selection_ratio=0.5, grid_size=(300,300)):
        """
        Args:
            dir: base path that contains the data
            samples_per_grid: number of randomly selected samples per grid
            selection_ratio: the ratio of samples to be selected per iteration 
                over the dataset with respect to all samples
        """
        super().__init__()
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.total_grids = len(self.files)

        self.samples_per_grid = samples_per_grid
        self.selection_ratio = selection_ratio
        self.total_samples_per_grid = grid_size[0] * grid_size[1]
        self.selection_per_file = int(self.total_samples_per_grid * self.selection_ratio)

    def __len__(self):
        total_samples = self.total_grids * self.total_samples_per_grid
        return int(total_samples * self.selection_ratio)
    

    def index_to_file_index(self, index):
        if (index >= 0):
            return index // self.selection_per_file
        
        return self.index_to_file_index(self.__len__() + index)
    

    def __getitem__(self, index):
        if (isinstance(index, int)):
            index = self.index_to_file_index(index)

            grid = torch.from_numpy(np.load(f"{self.dir}{self.files[index]}"))

            random_coords = torch.randint(0, 300, (self.samples_per_grid, 2))
            x_indices = random_coords[:, 0]
            y_indices = random_coords[:, 1]

            sampled_grid = grid[:, x_indices, y_indices].T

            time = int(self.files[index].split(".")[0])
            time_tensor = torch.full_like(x_indices, time).unsqueeze(-1)

            inputs = torch.concat((random_coords, time_tensor), dim=1)

            return inputs, sampled_grid
        
        elif (isinstance(index, torch.Tensor)):
            print("shit")
            if (len(index.shape) > 1):
                raise ValueError(f"index tensor must be 1D, got {index.shape}")
            
            inputs, sampled_grids = zip(*(self.__getitem__(idx.item()) for idx in index))

            return torch.concatenate(inputs), torch.concatenate(sampled_grids)
        
        else:
            raise TypeError(f"{type(index)}")        




            




    