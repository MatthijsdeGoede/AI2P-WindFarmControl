import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from architecture.windspeedLSTM.windspeedLSTM import WindspeedLSTM
from experiments.futureWindspeedLSTM.WindspeedMapDataset import create_data_loaders
from utils.preprocessing import resize_windspeed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(case):
    return {
        "case": case,
        "root_dir": f"../../data/Case_0{case}/measurements_flow/postProcessing_BL/windspeedMapScalars",
        "sequence_length": 50,
        "batch_size": 4,
        "scale": (128, 128)
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

    train_loader, val_loader, test_loader = create_data_loaders(root_dir, sequence_length, batch_size,
                                                                transform=create_transform(scale))
    model = WindspeedLSTM(sequence_length, 300).to(device)

    output_folder = create_output_folder(case)

    save_config(output_folder, config)

    train(model, train_loader, val_loader, output_folder)


def train(model, train_loader, val_loader, output_folder):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = 100
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_losses = []
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_losses = []
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
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
