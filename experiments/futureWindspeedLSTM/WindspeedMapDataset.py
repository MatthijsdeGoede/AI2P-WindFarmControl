import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class WindspeedMapDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        self.len = len([name for name in os.listdir(root_dir) if name != "README.md"]) - 2 * sequence_length

    def __len__(self):
        return self.len


    def _get_sequence(self, start, transform):
        scalars = [np.load(f'{self.root_dir}/Windspeed_map_scalars_{30005 + (start + i) * 5}.npy') for i in range(self.sequence_length)]
        if transform:
            scalars = transform(scalars)
        scalars = torch.tensor(np.array(scalars))
        return scalars

    def __getitem__(self, idx):
        scalars = self._get_sequence(idx, self.transform)
        target_scalars = self._get_sequence(idx + self.sequence_length, self.target_transform)
        return scalars, target_scalars

def create_data_loaders(data_folder, sequence_length, batch_size, transform=None):
    dataset = WindspeedMapDataset(data_folder, sequence_length, transform=transform, target_transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    root_dir = "../../data/Case_01/measurements_flow/postProcessing_BL/windspeedMapScalars"
    dataset = WindspeedMapDataset(root_dir, 1)