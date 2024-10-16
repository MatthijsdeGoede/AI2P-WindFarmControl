import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class GridDataset(Dataset):
    def __init__(self, dir, 
                turbine_csv, 
                wind_csv, 
                data_type="no-wake", 
                wake_dir=None, 
                wake_turbine_csv=None, 
                only_grid_values=False, 
                sampling=False, 
                samples_per_grid=128, 
                top_vorticity=0.85):
        
        super().__init__()

        self.datasets = []
        self.data_type = data_type

        if (self.data_type == "no-wake"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, only_grid_values, sampling, samples_per_grid, top_vorticity))

        elif (self.data_type == "wake"):
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, only_grid_values, sampling, samples_per_grid, top_vorticity))

        elif (self.data_type == "both"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, only_grid_values, sampling, samples_per_grid, top_vorticity))
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, only_grid_values, sampling, samples_per_grid, top_vorticity))
            
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
    def __init__(self, dir, turbine_csv, wind_csv, only_grid_values=False, sampling=False, samples_per_grid=256, top_vorticity=0.8):
        super().__init__()

        self.dir = dir
        self.files = os.listdir(self.dir)

        try: 
            self.files.remove("README.md")
        except:
            pass

        self.only_grid_values = only_grid_values
        
        flat_index = torch.arange(90000)
        x_coords = flat_index % 300
        y_coords = flat_index // 300
        self.coords = torch.stack([x_coords, y_coords]).T
        self.coords = (self.coords - 149.5) / 149.5 #normalizing to -1, 1

        self.df_turbines = pd.read_csv(turbine_csv, index_col=0)
        self.df_wind = pd.read_csv(wind_csv, index_col=0)

        self.sampling = sampling
        self.samples_per_grid = samples_per_grid
        self.top_vorticity = top_vorticity

    def __len__(self):
        return len(self.files) 

    def get_turbine_data(self, time):
        df = self.df_turbines
        time_instance = df[df['time'] == time][['speed', 'yaw_sin', "yaw_cos"]]
        return torch.from_numpy(time_instance.to_numpy().flatten()).unsqueeze(0)    

    def get_wind_data(self, time):
        time_instance = self.df_wind[self.df_wind['time'] == time][['winddir_sin', 'winddir_cos']]
        return torch.from_numpy(time_instance.to_numpy()) 
    

    def _calculate_vorticity(self, flow_field):
        """
        Vorticity (ω) describes the local spinning motion of a fluid.
        ωx: dw/dy - dv/dw
        ωy: du/dz - dw/dx
        ωz: dv/dx - du/dy
        |ω|: sqrt(ωx**2 + ωy**2 + ωz**2)
        """
        du = torch.gradient(flow_field[:, :, 0])
        dv = torch.gradient(flow_field[:, :, 1])
        dw = torch.gradient(flow_field[:, :, 2])

        dudx, dudy = du[0], du[1]
        dvdx, dvdy = dv[0], dv[1]
        dwdx, dwdy = dw[0], dw[1]

        x_vorticity = dwdy
        y_vorticity = -dwdx
        z_vorticity = dvdx - dudy

        vorticity_intensity = torch.sqrt(x_vorticity ** 2 + y_vorticity **2 + z_vorticity ** 2)
        return vorticity_intensity


    def _get_sampled_indices(self, vorticity: torch.Tensor):
        """
        The model struggles the most with high vorticity areas. We assign a
        higher selection ratio to the top 20% vorticity areas for sampling.
        As a result with a sample size of 256, 128 of them should be from the
        top %20 vorticity and 128 of them should be from the rest of grid.
        """
        selection_likelihood = self.top_vorticity / (1 - self.top_vorticity) 

        vorticity = vorticity.flatten()
        threshold = torch.quantile(vorticity, self.top_vorticity)
        selection_odds = np.array(torch.where(vorticity < threshold, 1, selection_likelihood))

        indices = np.random.choice(range(torch.numel(vorticity)), 
                                   size=self.samples_per_grid, 
                                   replace=False, 
                                   p=selection_odds/selection_odds.sum())
        
        return list(indices)
        

    def __getitem__(self, index):
        if (isinstance(index, int) or isinstance(index, np.int32)):
            #3x300x300 to 300x300x3
            flow_field = torch.from_numpy(np.load(f"{self.dir}{self.files[index]}")).permute(1, 2, 0)

            if (self.sampling):
                vorticity = self._calculate_vorticity(flow_field)
                indices = self._get_sampled_indices(vorticity)

            flow_field = flow_field.flatten(0, 1) #90000x3

            time = int(self.files[index].split(".")[0])
            time_tensor = torch.full((self.coords.shape[0], 1), time)

            if (self.only_grid_values):
                inputs = torch.concat((self.coords, time_tensor), dim=1)
                if (self.sampling):
                    return inputs[indices], flow_field[indices]
                
                return inputs, flow_field

            turbine_data = self.get_turbine_data(time)
            turbine_data = turbine_data.repeat(time_tensor.shape[0], 1)

            wind_data = self.get_wind_data(time)
            wind_data = wind_data.repeat(time_tensor.shape[0], 1)

            inputs = torch.concat((self.coords, time_tensor, turbine_data, wind_data), dim=1)

            if (self.sampling):
                return inputs[indices], flow_field[indices]

            return inputs, flow_field        

        else:
            raise TypeError(f"{type(index)}")

