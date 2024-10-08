import torch

import numpy as np
import torch.nn as nn

from adamp import AdamP
from torch.optim import Adam
from box import Box
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from architecture.pignn.pignn import FlowPIGNN
from architecture.pignn.deconv import FCDeConvNet

import os
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data_range = range(30005, 42000 + 1, 5)
        self.root = root
        self.graph_paths = self.load_graph_paths()

    def load_graph_paths(self):
        graph_paths = [f"{self.root}/graph_{i}.pt" for i in self.data_range]
        return graph_paths

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        graph_path = self.graph_paths[idx]
        graph_data = torch.load(graph_path)
        return graph_data


class GraphTemporalDataset(Dataset):
    def __init__(self, root, seq_length, transform=None, pre_transform=None):
        super(GraphTemporalDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.seq_length = seq_length

    def _get_sequence(self, start):
        return [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for i in range(self.seq_length)]

    def len(self):
        return len([name for name in os.listdir(self.root)]) - 2 * self.seq_length

    def get(self, idx):
        return self._get_sequence(idx), self._get_sequence(idx + self.seq_length)


def custom_collate_fn(batch):
    # Each batch consists of a list of sequences (each a list of graphs)
    return [Data.from_data_list(seq) for seq in batch]


def create_data_loaders(data_folder, batch_size, seq_length):
    dataset = GraphTemporalDataset(root=data_folder, seq_length=seq_length) if seq_length > 1 else GraphDataset(root=data_folder)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    collate = custom_collate_fn if seq_length > 1 else None

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)

    return train_loader, val_loader, test_loader


def train(model, train_params, train_loader, val_loader, output_folder):
    print(model)

    optimizer = AdamP(model.parameters(), lr=1e-3) if torch.cuda.is_available() else Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params.num_epochs
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_losses = []
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss = compute_loss(batch, criterion, model)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_losses = []
            for batch in val_loader:
                batch = batch.to(device)
                val_loss = compute_loss(batch, criterion, model)
                val_losses.append(val_loss.item())
            model.train()

        learning_rate = optimizer.param_groups[0]['lr']
        avg_val_loss = np.mean(val_losses)
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {np.mean(train_losses)}, validation loss: {avg_val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")

        # Check early stopping criterion
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params.early_stop_after:
            print(f'Early stopping at epoch {epoch}')
            break


def compute_loss(batch, criterion, model):
    if isinstance(model, FCDeConvNet):
        # If the model is not a Graph Neural Network, just concatenate everything
        x = batch.x.to(device)
        pos = batch.pos.to(device)
        edge_attr = batch.edge_attr.to(device)
        glob = batch.global_feats.to(device)
        batch_size = glob.size(0)

        x_cat = torch.cat((
            x.reshape(batch_size, -1),
            pos.reshape(batch_size, -1),
            edge_attr.reshape(batch_size, -1),
            glob.reshape(batch_size, -1)
        ), dim=-1).float()

        pred = model(x_cat)
        target = batch.y.to(device).reshape(-1, pred.size(1))
        loss = criterion(pred, target)
    else:
        nf = torch.cat((batch.x.to(device), batch.pos.to(device)), dim=-1).float()
        ef = batch.edge_attr.to(device).float()
        gf = batch.global_feats.to(device).float()
        pred = model(batch, nf, ef, gf)
        target = batch.y.to(device).reshape(-1, pred.size(1))
        loss = criterion(pred, target)
    return loss


def create_output_folder(train_config, net_type):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config.case_nr}_{train_config.wake_steering}_{net_type}__{train_config.seq_length}"
    os.makedirs(output_folder)
    return output_folder


def get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True, seq_length=1, num_epochs=200):
    cfg = Box({
        'model': {
            'edge_in_dim': 2,
            'node_in_dim': 5,
            'global_in_dim': 2,
            'n_pign_layers': 3,
            'edge_hidden_dim': 50,
            'node_hidden_dim': 50,
            'global_hidden_dim': 50,
            'num_nodes': 10,
            'residual': True,
            'input_norm': True,
            'pign_mlp_params': {
                'num_neurons': [256, 128],
                'hidden_act': 'ReLU',
                'out_act': 'ReLU'
            },
            'reg_mlp_params': {
                'num_neurons': [64, 128, 256],
                'hidden_act': 'ReLU',
                'out_act': 'ReLU'
            },
        },
        'train': {
            'case_nr': case_nr,
            'wake_steering': wake_steering,
            'max_angle': max_angle,
            'num_epochs': num_epochs,
            'use_graph': use_graph,
            'seq_length': seq_length,
            'early_stop_after': 5,
            'batch_size': 64,
        }
    })
    return cfg


def run_experiments(is_temporal):
    seq_length = 50 if is_temporal else 1
    experiment_cfgs = [
        get_config(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=False, seq_length=seq_length),

        get_config(case_nr=1, wake_steering=True, max_angle=30, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=True, max_angle=90, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=True, seq_length=seq_length),
        get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=False, seq_length=seq_length),

        # get_config(case_nr=2, wake_steering=False, max_angle=30, use_graph=True),
        # get_config(case_nr=2, wake_steering=False, max_angle=90, use_graph=True),
        # get_config(case_nr=2, wake_steering=False, max_angle=360, use_graph=True),
        # get_config(case_nr=2, wake_steering=False, max_angle=360, use_graph=False),
    ]

    for i, cfg in enumerate(experiment_cfgs):
        post_fix = "LuT2deg_internal" if cfg.train.wake_steering else "BL"
        net_type = f"pignn_deconv_{cfg.train.max_angle}" if cfg.train.use_graph else "fcn_deconv"
        data_folder = f"../../data/Case_0{cfg.train.case_nr}/graphs/{post_fix}/{cfg.train.max_angle}"
        output_folder = create_output_folder(cfg.train, net_type)

        train_loader, val_loader, test_loader = create_data_loaders(data_folder, cfg.train.batch_size, cfg.train.seq_length)
        graph_model = FlowPIGNN(**cfg.model).to(device) if cfg.train.use_graph else FCDeConvNet(232, 650, 656, 500).to(device)

        print(f"Running experiment {i+1}/{len(experiment_cfgs)}")
        if seq_length > 1:
            print("tada")
        # temporal_model = WindspeedLSTM(seq_length, 128).to(device)
        # train_temporal(graph_model, temporal_model, config, train_loader, val_loader, output_folder)
        else:
            train(graph_model, cfg.train, train_loader, val_loader, output_folder)


if __name__ == "__main__":
    temporal = False
    run_experiments(temporal)
