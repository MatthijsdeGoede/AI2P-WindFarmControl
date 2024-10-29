import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


L = 5000 #Characterestic length in meters. Choosen 5000 because it's the simulation area
U = 8 #Characteristic velocity in meters per second. Choosen by assuming the free stream velocity is 8 m/s
HUB_HEIGHT = 119 #Height of the turbine in meters


class GridDataset(Dataset):
    def __init__(self, dir, 
                turbine_csv, 
                wind_csv, 
                data_type="wake", 
                wake_dir=None, 
                wake_turbine_csv=None, 
                sampling=False, 
                samples_per_grid=128, 
                top_vorticity=0.80,
                time_scaling_factor = 1.0):
        
        super().__init__()

        self.datasets = []
        self.data_type = data_type

        self.time_scaling_factor = time_scaling_factor

        if (self.data_type == "no-wake"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, sampling, samples_per_grid, top_vorticity))

        elif (self.data_type == "wake"):
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, sampling, samples_per_grid, top_vorticity))

        elif (self.data_type == "both"):
            self.datasets.append(CaseDataset(dir, turbine_csv, wind_csv, sampling, samples_per_grid, top_vorticity))
            self.datasets.append(CaseDataset(wake_dir, wake_turbine_csv, wind_csv, sampling, samples_per_grid, top_vorticity))
            
        else:
            raise ValueError(f"data_type must be no-wake, wake, or both. was given: {self.data_type}")


    def __len__(self):
        return sum([len(x) for x in self.datasets])
    

    def _non_dimensionalize_inputs(self, inputs: torch.Tensor):
        """
        The first two columns of the input is x-y coordinates of a 300x300 grid. The original simulation is
        performed over a 5000m x 5000m area. First let's convert the grid indices to distances and then non dimensionalize it.

        Third column is the hub height. However, this is in meters already so we don't have convert it, only non dimensionalize it.

        The fourth column is time in seconds. The simulation starts at time = 30.000s and continues for 12.000 seconds

        The inputs are scaled to (-x, x).
        """
        inputs_ = inputs.detach().clone()

        inputs_[:, 0] = inputs_[:, 0] * (5000/300)
        inputs_[:, 1] = inputs_[:, 1] * (5000/300)
        inputs_[:, 3] = inputs_[:, 3] - 30000

        inputs_[:, :3] = inputs_[:, :3] / L
        inputs_[:, 3] = inputs_[:, 3] * U / L
        
        mean_distance = (5000 / L) / 2
        mean_time = (12000 * U / L) / 2

        inputs_[:, :3] = inputs_[:, :3] - mean_distance
        inputs_[:, 3] = (inputs_[:, 3] - mean_time) * self.time_scaling_factor

        return inputs_
    
    def _non_dimensionalize_targets(self, targets: torch.Tensor):
        targets_ = targets.detach().clone()
        targets_ = targets_ / U
        return targets_
    

    def is_index_wake_steering(self, index):    
        files_per_case = len(self.datasets[0])
        is_wake_steering = False if (index // files_per_case == 0) else True
        index = index % files_per_case

        return is_wake_steering, index

    def __getitem__(self, index):
        if (self.data_type != "both"):
            out =  self.datasets[0][index]
        
        else:
            is_wake_steering, index = self.is_index_wake_steering(index)
            out = self.datasets[1][index] if is_wake_steering else self.datasets[0][index]

        return self._non_dimensionalize_inputs(out[0]), self._non_dimensionalize_targets(out[1])


class CaseDataset(Dataset):
    def __init__(self, dir, turbine_csv, wind_csv, sampling=False, samples_per_grid=256, top_vorticity=0.8):
        super().__init__()

        self.dir = dir
        self.files = os.listdir(self.dir)

        try: 
            self.files.remove("README.md")
        except:
            pass
        
        flat_index = torch.arange(90000)
        x_coords = flat_index % 300
        y_coords = flat_index // 300
        z_coords = torch.full((90000,), HUB_HEIGHT)
        self.coords = torch.stack([x_coords, y_coords, z_coords]).T

        self.df_turbines = pd.read_csv(turbine_csv, index_col=0)
        self.df_wind = pd.read_csv(wind_csv, index_col=0)

        self.sampling = sampling
        self.samples_per_grid = samples_per_grid
        self.top_vorticity = top_vorticity

    def __len__(self):
        return len(self.files) 

    def get_turbine_data(self, time):
        df = self.df_turbines
        time_instance = df[df['time'] == time][['yaw_sin', "yaw_cos"]]
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

