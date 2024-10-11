import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from architecture.UNetGraphLSTM.UNetGraphLSTM import UNetGraphLSTM
from experiments.UNetGraphLSTM.UnetGrapLSTMDataset import create_data_loaders
from experiments.graphs.graph_experiments import get_config
from utils.preprocessing import resize_windspeed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def load_config(case):
    return {
        "case": case,
        "root_dir": f"../../data/Case_0{case}/measurements_flow/postProcessing_BL/windspeedMapScalars",
        "sequence_length": 50,
        "batch_size": 4,
        "scale": (128,128)
    }


def create_transform(scale):
    def resize_scalars(windspeed_scalars):
        return [resize_windspeed(scalar, scale) for scalar in windspeed_scalars]
    return resize_scalars


def run():
    case = 1
    config = load_config(case)

    root_dir = config["root_dir"]
    sequence_length = config["sequence_length"]
    batch_size = config["batch_size"]
    scale = config["scale"]

    graph_path = f"../../data/Case_0{case}/graphs/BL/90"
    windspeeds_path = root_dir

    train_loader, val_loader, test_loader = create_data_loaders(graph_path, windspeeds_path, sequence_length, batch_size,
                                                                transform_windspeeds=create_transform(scale))
    model = UNetGraphLSTM(sequence_length=sequence_length, input_size=300,
                          **get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True).model).to(device)

    output_folder = create_output_folder(case)

    save_config(output_folder, config)

    train(model, train_loader, val_loader, output_folder)

def compute_loss(input_graphs, input_windspeeds, targets, criterion, model):
    nf = torch.cat((input_graphs.x.to(device), input_graphs.pos.to(device)), dim=-1)
    ef = input_graphs.edge_attr.to(device)
    gf = input_graphs.global_feats.to(device)
    outputs = model(input_windspeeds, input_graphs, nf, ef, gf)
    loss = criterion(outputs, targets)
    return loss

def train(model, train_loader, val_loader, output_folder):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = 100
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_losses = []
        for idx, (input_graphs, input_windspeeds, targets) in enumerate(train_loader):
            input_graphs = input_graphs.to(device)
            input_windspeeds = input_windspeeds.to(device)
            targets = targets.to(device)
            loss = compute_loss(input_graphs, input_windspeeds, targets, criterion, model)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_losses = []
            for input_graphs, input_windspeeds, targets in val_loader:
                input_graphs = input_graphs.to(device)
                input_windspeeds = input_windspeeds.to(device)
                targets = targets.to(device)
                val_loss = compute_loss(input_graphs, input_windspeeds, targets, criterion, model)
                val_losses.append(val_loss.item())
            model.train()

        learning_rate = optimizer.param_groups[0]['lr']
        avg_val_loss = np.mean(val_losses)
        print(
            f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {np.mean(train_losses)}, validation loss: {avg_val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/model/{epoch}.pt")

        # Check early stopping criterion
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 5:
            print(f'Early stopping at epoch {epoch}')
            break

def create_output_folder(case_nr):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{case_nr}"
    os.makedirs(f"{output_folder}/model")
    return output_folder

def save_config(output_folder, config):
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    run()