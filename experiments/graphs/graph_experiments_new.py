import numpy as np
import torch
import torch.nn as nn
from box import Box
from torch.optim import Adam
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from architecture.pignn.pignn import FlowPIGNN
from architecture.pignn.deconv import FCDeConvNet
from datetime import datetime
import os

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


def compute_loss(batch, criterion, model):
    # Logic to handle different model types
    x, pos, edge_attr, glob, target = batch.x, batch.pos, batch.edge_attr, batch.global_feats, batch.y
    # Concatenate features for non-GNN models
    if isinstance(model, FCDeConvNet):
        x_cat = torch.cat([x.flatten(), pos.flatten(), edge_attr.flatten(), glob.flatten()], dim=-1)
        pred = model(x_cat)
    else:
        nf = torch.cat((x, pos), dim=-1)
        pred = model(batch, nf, edge_attr, glob)
    loss = criterion(pred, target.reshape((pred.size(0), -1)))
    return loss


def train(model, train_loader, val_loader, config):
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(config.train.num_epochs):
        train_losses, val_losses = [], []
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(batch, criterion, model)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss = compute_loss(batch, criterion, model)
                val_losses.append(val_loss.item())

        print(f"Epoch {epoch}: Train Loss: {np.mean(train_losses)}, Val Loss: {np.mean(val_losses)}")

        # Save model pointer
        torch.save(model.state_dict(), f"{config.output_folder}/pignn_{epoch}.pt")

        # Check early stopping criterion
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{config.output_folder}/best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.train.early_stop_after:
            print(f'Early stopping at epoch {epoch}')
            break


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
        },
        'data': {
            'input':  f"../../data/Case_0{case_nr}/graphs/{'LuT2deg_internal' if wake_steering else 'BL'}/{max_angle}"
        }
    })
    return cfg


def run_experiment(config):
    train_loader, val_loader, _ = create_data_loaders(config.data.input, config.train.batch_size, config.train.seq_length)
    graph_model = FlowPIGNN(**config.model).to(device) if config.train.use_graph else FCDeConvNet(232, 650, 656, 500).to(device)
    output_folder = create_output_folder(config.train)
    config.output_folder = output_folder
    seq_length = config.train.seq_length

    if seq_length > 1:
        temporal_model = WindspeedLSTM(seq_length, 128).to(device)
        # train_temporal(graph_model, temporal_model, config, train_loader, val_loader, output_folder)
    else:
        train(graph_model, train_loader, val_loader, config)


def create_output_folder(t_cfg):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    net_type = f"pignn_deconv_{t_cfg.max_angle}" if t_cfg.use_graph else "fcn_deconv"
    output_folder = f"results/{timestamp}_Case{t_cfg.case_nr}_{t_cfg.wake_steering}_{net_type}_{t_cfg.seq_length}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run experiments with different configurations.')
    # parser.add_argument('--case_nr', type=int, default=1, help='Case number to use for the experiment (default: 1)')
    # parser.add_argument('--wake_steering', action='store_true', help='Enable wake steering (default: False)')
    # parser.add_argument('--max_angle', type=int, default=90, help='Maximum angle for the experiment (default: 90)')
    # parser.add_argument('--use_graph', action='store_true', help='Use graph representation (default: False)')
    # parser.add_argument('--seq_length', type=int, default=1, help='Sequence length for the experiment (default: 1)')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the experiment (default: 4)')
    # args = parser.parse_args()
    # run(args.case_nr, args.wake_steering, args.max_angle, args.use_graph, args.seq_length, args.batch_size)

    run_experiment(get_config(1, False, 30, True, 1))