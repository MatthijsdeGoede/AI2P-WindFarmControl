import json
import multiprocessing

import torch

import numpy as np
import torch.nn as nn

from adamp import AdamP
from torch.optim import Adam
from box import Box
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from architecture.pignn.pignn import FlowPIGNN
from architecture.pignn.deconv import FCDeConvNet

import os
from datetime import datetime

from architecture.windspeedLSTM.windspeedLSTM import WindspeedLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.graph_paths = self.load_graph_paths()

    def load_graph_paths(self):
        graph_paths = [f"{self.root}/graph_{i}.pt" for i in range(30005, 42000 + 1, 5)]
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


def create_data_loaders(dataset, batch_size, custom_collate=None):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    num_workers = multiprocessing.cpu_count()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)

    return train_loader, val_loader, test_loader


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


def train_epoch(train_loader, model, criterion, optimizer, scheduler):
    train_losses = []
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss = compute_loss(batch, criterion, model)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_epoch(val_loader, model, criterion):
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            batch = batch.to(device)
            val_loss = compute_loss(batch, criterion, model)
            val_losses.append(val_loss.item())
    model.train()
    return np.mean(val_losses)


def train(model, train_params, train_loader, val_loader, output_folder):
    optimizer = AdamP(model.parameters(), lr=1e-3) if torch.cuda.is_available() else Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params.num_epochs
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss = eval_epoch(val_loader, model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")

        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params.early_stop_after:
            print(f'Early stopping at epoch {epoch}')
            break


def process_temporal_batch(batch, graph_model, temporal_model, criterion):
    generated_img = []
    target_img = []
    for i, seq in enumerate(batch[0]):
        # Process graphs in parallel at each timestep for the entire batch
        seq = seq.to(device)
        nf = torch.cat((seq.x.to(device), seq.pos.to(device)), dim=-1)
        ef = seq.edge_attr.to(device)
        gf = seq.global_feats.to(device)
        graph_output = graph_model(seq, nf, ef, gf).reshape(-1, 128, 128)
        generated_img.append(graph_output)
        target_img.append(batch[1][i].y.to(device).reshape(-1, 128, 128))

    temporal_img = torch.stack(generated_img, dim=1)
    output = temporal_model(temporal_img).float()
    target = torch.stack(target_img, dim=1)
    return criterion(output, target)


def train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler):
    train_losses = []
    for i, batch in enumerate(train_loader):
        print(f"processing batch {i + 1}/{len(train_loader)}")
        loss = process_temporal_batch(batch, graph_model, temporal_model, criterion)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion):
    with torch.no_grad():
        graph_model.eval()
        temporal_model.eval()
        val_losses = []
        for batch in val_loader:
            val_loss = process_temporal_batch(batch, graph_model, temporal_model, criterion)
            val_losses.append(val_loss.item())
    graph_model.train()
    temporal_model.train()
    return np.mean(val_losses)


def train_temporal(graph_model, temporal_model, train_params, train_loader, val_loader, output_folder):
    optimizer = torch.optim.Adam(list(graph_model.parameters()) + list(temporal_model.parameters()), lr=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params.num_epochs
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler)
        val_loss = eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointers
        torch.save(graph_model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")
        torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_{epoch}.pt")

        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params.early_stop_after:
            print(f'Early stopping at epoch {epoch}')
            break


def create_output_folder(train_config, net_type):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config.case_nr}_{train_config.wake_steering}_{net_type}_" \
                    f"{train_config.seq_length}"
    os.makedirs(output_folder)
    return output_folder


def save_config(output_folder, config):
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)


def get_pignn_config():
    return {
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
    }


def get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True, num_epochs=100, seq_length=1,
               early_stop_after=5, batch_size=4):
    cfg = Box({
        'model': get_pignn_config(),
        'train': {
            'case_nr': case_nr,
            'wake_steering': wake_steering,
            'max_angle': max_angle,
            'num_epochs': num_epochs,
            'use_graph': use_graph,
            'early_stop_after': early_stop_after,
            'batch_size': batch_size,
            'seq_length': seq_length,
        }
    })
    return cfg


def run_experiments():
    experiment_cfgs = [
        get_config(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=50),
        # get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True),
        # get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=True),
        # get_config(case_nr=1, wake_steering=False, max_angle=360, use_graph=False),
        #
        # get_config(case_nr=1, wake_steering=True, max_angle=30, use_graph=True),
        # get_config(case_nr=1, wake_steering=True, max_angle=90, use_graph=True),
        # get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=True),
        # get_config(case_nr=1, wake_steering=True, max_angle=360, use_graph=False),
        #
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
        save_config(output_folder, cfg)
        seq_length = cfg.train.seq_length

        dataset = GraphTemporalDataset(root=data_folder, seq_length=seq_length) if seq_length > 1 else GraphDataset(root=data_folder)
        collate = custom_collate_fn if seq_length > 1 else None
        train_loader, val_loader, test_loader = create_data_loaders(dataset, cfg.train.batch_size, collate)
        graph_model = FlowPIGNN(**cfg.model).to(device) if cfg.train.use_graph else FCDeConvNet(232, 650, 656, 500).to(device)

        if seq_length > 1:
            temporal_model = WindspeedLSTM(seq_length, 128).to(device)
            train_temporal(graph_model, temporal_model, cfg.train, train_loader, val_loader, output_folder)
        else:
            train(graph_model, cfg.train, train_loader, val_loader, output_folder)

        print(f"Running experiment {i + 1}/{len(experiment_cfgs)}")


if __name__ == "__main__":
    run_experiments()
