import os
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from experiments.futureWindspeedLSTM.WindspeedMapDataset import WindspeedMapDataset
from experiments.graphs.GraphDataset import GraphDataset


class UnetGraphLSTMDataset(Dataset):
    def __init__(self, graph_path, windspeeds_path, sequence_length,
                 transform_graph=None, target_transform_graph=None,
                 transform_windspeed=None, target_transform_windspeed=None):
        super(UnetGraphLSTMDataset, self).__init__()
        self.graphs = GraphDataset(graph_path,
                                   transform_graph, target_transform_graph)
        self.windspeeds = WindspeedMapDataset(windspeeds_path, sequence_length,
                                              transform_windspeed, target_transform_windspeed)
        self.sequence_length = sequence_length

    def __len__(self):
        return min(len(self.graphs), len(self.windspeeds))

    def __getitem__(self, idx):
        windspeeds, targets = self.windspeeds[idx]
        graphs = self.graphs[idx + self.sequence_length]
        return graphs, windspeeds, targets

def create_data_loaders(graph_path, windspeeds_path, sequence_length, batch_size, transform_windspeeds=None):
    dataset = UnetGraphLSTMDataset(graph_path, windspeeds_path, sequence_length,
                                   transform_windspeed=transform_windspeeds, target_transform_windspeed=transform_windspeeds)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2], generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    return train_loader, val_loader, test_loader