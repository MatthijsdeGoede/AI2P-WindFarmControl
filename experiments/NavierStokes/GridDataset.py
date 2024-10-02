import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class GridDataset(Dataset):
    def __init__(self, dir, turbine_csv, wind_csv, data_type="no-wake", wake_dir=None, wake_turbine_csv=None, only_grid_values=False):
        super().__init__()

        self.datasets = []
        self.data_type = data_type

        if (self.data_type == "no-wake"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, only_grid_values))

        elif (self.data_type == "wake"):
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, only_grid_values))

        elif (self.data_type == "both"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, only_grid_values))
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, only_grid_values))
            
        else:
            raise ValueError(f"data_type must be no-wake, wake, or both. was given: {self.data_type}")


    def __len__(self):
        return sum([len(x) for x in self.datasets])
    

    def is_index_wake_steering(self, index):    
        files_per_case = len(self.datasets[0])
        is_wake_steering = False if (index // files_per_case == 0) else True
        index = index % files_per_case

        return is_wake_steering, index

    def __getitem__(self, index):
        if (self.data_type != "both"):
            return self.datasets[0][index]
        
        else:
            is_wake_steering, index = self.is_index_wake_steering(index)
            return self.datasets[1][index] if is_wake_steering else self.datasets[0][index]

    

class CaseDataset(Dataset):
    def __init__(self, dir, turbine_csv, wind_csv, only_grid_values=False):
        super().__init__()

        self.dir = dir
        self.files = os.listdir(self.dir)

        try: 
            self.files.remove("README.md")
        except:
            pass

        self.only_grid_values = only_grid_values
        
        flat_index = torch.arange(90000)
        x_coords = flat_index // 300
        y_coords = flat_index % 300
        self.coords = torch.stack([x_coords, y_coords]).T

        self.df_turbines = pd.read_csv(turbine_csv, index_col=0)
        self.df_wind = pd.read_csv(wind_csv, index_col=0)       

    def __len__(self):
        return len(self.files) 

    def get_turbine_data(self, time):
        df = self.df_turbines
        time_instance = df[df['time'] == time][['speed', 'yaw_sin', "yaw_cos"]]
        return torch.from_numpy(time_instance.to_numpy().flatten()).unsqueeze(0)    

    def get_wind_data(self, time):
        time_instance = self.df_wind[self.df_wind['time'] == time][['winddir_sin', 'winddir_cos']]
        return torch.from_numpy(time_instance.to_numpy())
    

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




            




    