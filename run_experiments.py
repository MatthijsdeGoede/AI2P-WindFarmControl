import torch

import numpy as np
import torch.nn as nn

from adamp import AdamP
from torch.optim import Adam
from box import Box
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from architecture.nets.pignn import FlowPIGNN
from architecture.nets.deconv import FCDeConvNet

import os
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphDataset(Dataset):
    def __init__(self, root, data_range, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data_range = data_range
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


def create_data_loaders(data_folder, batch_size):
    dataset = GraphDataset(root=data_folder, data_range=range(30005, 42000 + 1, 5))
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
        loss = criterion(pred, batch.y.reshape(-1, pred.size(1)))
    else:
        nf = torch.cat((batch.x.to(device), batch.pos.to(device)), dim=-1).float()
        ef = batch.edge_attr.to(device).float()
        gf = batch.global_feats.to(device).float()
        pred = model(batch, nf, ef, gf)
        loss = criterion(pred, batch.y.reshape(-1, pred.size(1)))
    return loss


def create_output_folder(train_config, net_type):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config.case_nr}_{train_config.wake_steering}_{net_type}"
    os.makedirs(output_folder)
    return output_folder


def get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True, num_epochs=200):
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
            'early_stop_after': 5,
            'batch_size': 64,
        }
    })
    return cfg


def run_experiments():
    experiment_cfgs = [
        get_config(case_nr=1, wake_steering=False, max_angle=30, use_graph=True),
        get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True),
        get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=True),
        get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=False),

        get_config(case_nr=1, wake_steering=True, max_angle=30, use_graph=True),
        get_config(case_nr=1, wake_steering=True, max_angle=90, use_graph=True),
        get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=True),
        get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=False),

        get_config(case_nr=2, wake_steering=False, max_angle=30, use_graph=True),
        get_config(case_nr=2, wake_steering=False, max_angle=90, use_graph=True),
        get_config(case_nr=2, wake_steering=False, max_angle=360, use_graph=True),
        get_config(case_nr=2, wake_steering=False, max_angle=360, use_graph=False),
    ]

    for i, cfg in enumerate(experiment_cfgs):
        post_fix = "LuT2deg_internal" if cfg.train.wake_steering else "BL"
        net_type = f"pignn_deconv_{cfg.train.max_angle}" if cfg.train.use_graph else "fcn_deconv"
        data_folder = f"D:/AI2P/data/Case_0{cfg.train.case_nr}/graphs/{post_fix}/{cfg.train.max_angle}"
        output_folder = create_output_folder(cfg.train, net_type)

        train_loader, val_loader, test_loader = create_data_loaders(data_folder, cfg.train.batch_size)
        model = FlowPIGNN(**cfg.model).to(device) if cfg.train.use_graph else FCDeConvNet(232, 650, 656, 500).to(device)

        print(f"Running experiment {i+1}/{len(experiment_cfgs)}")
        train(model, cfg.train, train_loader, val_loader, output_folder)


# TODO: prepare loader for temporal network
if __name__ == "__main__":
    run_experiments()
