import argparse
import json
import os
import torch

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from architecture.WGN.pignn import FlowPIGNN
from architecture.WGN.deconv import FCDeConvNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WGNDataset(Dataset):
    def __init__(self, data_folder, max_angle):
        self.data_folder = data_folder
        self.max_angle = max_angle
        # Get list of all files
        self.dataset = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the data, e.g., an image or tensor from file
        grid_path = self.dataset[idx]
        graph_data = torch.load(grid_path, weights_only=False)

        return graph_data

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
    def __init__(self, root, seq_length, preload=True, transform=None, pre_transform=None):
        super(GraphTemporalDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.seq_length = seq_length
        self.preload = preload

        if preload:
            self.data = [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for start in range(self.len()) for i in range(self.seq_length)]

    def _get_sequence(self, start):
        if self.preload:
            return [self.data[start + i] for i in range(self.seq_length)]
        else:
            return [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for i in range(self.seq_length)]

    def len(self):
        return len([name for name in os.listdir(self.root)]) - 2 * self.seq_length

    def get(self, idx):
        return self._get_sequence(idx), self._get_sequence(idx + self.seq_length)


def custom_collate_fn(batch):
    # Each batch consists of a list of sequences (each a list of graphs)
    return [Data.from_data_list(seq) for seq in batch]


def create_data_loaders(dataset, batch_size, seq_length):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    collate = custom_collate_fn if seq_length > 1 else None
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_loss(batch, criterion, model):
    # Logic to handle different model types
    x, pos, edge_attr, glob, target = batch.x, batch.pos, batch.edge_attr.float(), batch.global_feats.float(), batch.y
    # Concatenate features for non-GNN models
    if isinstance(model, FCDeConvNet):
        x_cat = torch.cat([x.flatten(), pos.flatten(), edge_attr.flatten(), glob.flatten()], dim=-1).float()
        pred = model(x_cat)
    else:
        nf = torch.cat((x, pos), dim=-1).float()
        pred = model(batch, nf, edge_attr, glob)
    loss = criterion(pred, target.reshape((pred.size(0), -1, 128, 128)))
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
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss = eval_epoch(val_loader, model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/WGN_{epoch}.pt")
        if epoch==num_epochs:
            np.save("train_data/train_loss", train_loss_list)
            np.save("train_data/val_loss", val_loss_list)
        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{output_folder}/WGN_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            print(f'Early stopping at epoch {epoch}')
            np.save("train_data/train_loss", train_loss_list)
            np.save("train_data/val_loss", val_loss_list)
            break


def process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size):
    generated_img = []
    target_img = []
    for i, seq in enumerate(batch[0]):
        # Process graphs in parallel at each timestep for the entire batch
        seq = seq.to(device)
        nf = torch.cat((seq.x.to(device), seq.pos.to(device)), dim=-1).float()
        ef = seq.edge_attr.to(device).float()
        gf = seq.global_feats.to(device).float()
        graph_output = graph_model(seq, nf, ef, gf).reshape(-1, embedding_size[0], embedding_size[1])
        generated_img.append(graph_output)
        target_img.append(batch[1][i].y.to(device).reshape(-1, 128, 128))

    temporal_img = torch.stack(generated_img, dim=1)
    output = temporal_model(temporal_img)
    target = torch.stack(target_img, dim=1)
    return criterion(output, target)


def train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler, embedding_size):
    train_losses = []
    for i, batch in enumerate(train_loader):
        print(f"processing batch {i + 1}/{len(train_loader)}")
        loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size):
    with torch.no_grad():
        graph_model.eval()
        temporal_model.eval()
        val_losses = []
        for batch in val_loader:
            val_loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size)
            val_losses.append(val_loss.item())
    graph_model.train()
    temporal_model.train()
    return np.mean(val_losses)


def train_temporal(graph_model, temporal_model, train_params, train_loader, val_loader, output_folder, embedding_size):
    optimizer = Adam(list(graph_model.parameters()) + list(temporal_model.parameters()), lr=0.01)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler, embedding_size)
        val_loss = eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size)
        learning_rate = optimizer.param_groups[0]['lr']
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointers
        torch.save(graph_model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")
        torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_{epoch}.pt")

        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(graph_model.state_dict(), f"{output_folder}/pignn_best.pt")
            torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            print(f'Early stopping at epoch {epoch}')
            break


def create_output_folder(train_config, net_type):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config['case_nr']}_{train_config['wake_steering']}_{net_type}_" \
                    f"{train_config['seq_length']}"
    os.makedirs(output_folder)
    return output_folder


def save_config(output_folder, config):
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)


def get_pignn_config():
    return {
        'edge_in_dim': 2,
        'node_in_dim': 4,
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


def get_config(case_nr, wake_steering, max_angle, use_graph, seq_length, batch_size, output_size, direct_lstm=False, num_epochs=200, early_stop_after=10):
    return get_pignn_config(), {
        'case_nr': case_nr,
        'wake_steering': wake_steering,
        'max_angle': max_angle,
        'num_epochs': num_epochs,
        'use_graph': use_graph,
        'early_stop_after': early_stop_after,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'direct_lstm': direct_lstm,
        'output_size': output_size,
    }


def run(case_nr, wake_steering, max_angle, use_graph, seq_length, batch_size, direct_lstm, output_size):
    model_cfg, train_cfg = get_config(case_nr, wake_steering, max_angle, use_graph, seq_length, batch_size, output_size, direct_lstm)
    is_direct_lstm = train_cfg['direct_lstm']
    is_temporal = seq_length > 1
    post_fix = "LuT2deg_internal" if train_cfg['wake_steering'] else "BL"
    pignn_type = ("pignn_lstm_deconv" if is_direct_lstm else "pignn_unet_lstm") if is_temporal else "pignn_deconv"
    net_type = f"{pignn_type}_{train_cfg['max_angle']}" if train_cfg['use_graph'] else "fcn_deconv"

    data_folder = f"../../data/Case_0{train_cfg['case_nr']}/graphs/{post_fix}/{train_cfg['max_angle']}"
    output_folder = create_output_folder(train_cfg, net_type)
    save_config(output_folder, train_cfg)

    dataset = GraphTemporalDataset(root=data_folder, seq_length=seq_length) if is_temporal else GraphDataset(root=data_folder)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, train_cfg['batch_size'], seq_length)

    graph_model = FlowPIGNN(**model_cfg, output_size=(output_size, output_size)).to(device) if train_cfg['use_graph'] else FCDeConvNet(232, 650, 656, 500).to(device)

    train(graph_model, train_cfg, train_loader, val_loader, output_folder)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run experiments with different configurations.')
    # parser.add_argument('--case_nr', type=int, default=1, help='Case number to use for the experiment (default: 1)')
    # parser.add_argument('--wake_steering', action='store_true', help='Enable wake steering (default: False)')
    # parser.add_argument('--max_angle', type=int, default=90, help='Maximum angle for the experiment (default: 90)')
    # parser.add_argument('--use_graph', action='store_true', help='Use graph representation (default: False)')
    # parser.add_argument('--seq_length', type=int, default=1, help='Sequence length for the experiment (default: 1)')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the experiment (default: 4)')
    # parser.add_argument('--direct_lstm', action='store_true', help='Feed the PIGNN output directly to the LSTM (default: False)')
    # args = parser.parse_args()
    # run(args.case_nr, args.wake_steering, args.max_angle, args.use_graph, args.seq_length, args.batch_size, args.direct_lstm)

    run(1, True, 30, True, 1, 64, False, 300)
    # run(1, False, 90, True, 1, 64, False)
    # run(1, False, 360, True, 1, 64, False)
    # run(1, False, 360, False, 1, 64, False)

    # run(1, True, 30, True, 1, 64, False)
    # run(1, True, 90, True, 1, 64, False)
    # run(1, True, 360, True, 1, 64, False)
    # run(1, True, 360, False, 1, 64, False)
