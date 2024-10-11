import os

import torch
from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.graph_paths = self.load_graph_paths()

    def load_graph_paths(self):
        graph_paths = [f"{self.root}/{file}" for file in os.listdir(self.root)]
        return graph_paths

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        graph_path = self.graph_paths[idx]
        graph_data = torch.load(graph_path)
        return graph_data