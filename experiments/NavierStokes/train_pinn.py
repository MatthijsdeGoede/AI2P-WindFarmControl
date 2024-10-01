import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm

from PINN import PINN
from GridDataset import GridDataset
from IncNSLoss import NSLoss

assert torch.cuda.is_available()

generator = torch.Generator()
generator.manual_seed(42)

def main(physics_coef: int, hidden_size: int, only_grid: bool, use_wake: bool, model_save_path: str):
    dataset = GridDataset(dir="data/Case_01/measurements_flow/postProcessing_BL/winSpeedMapVector/",
                        turbine_csv="data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv",
                        wind_csv="data/Case_01/winDir_processed.csv", 
                        use_wake=use_wake, 
                        wake_dir="data/Case_01/measurements_flow/postProcessing_LuT2deg_internal/winSpeedMapVector/",
                        wake_turbine_csv="data/Case_01/measurements_turbines/30000_LuT2deg_internal/rot_yaw_combined.csv",
                        only_grid_values=only_grid)

    MIN_TIME, MAX_TIME = dataset[0][0][:, 2][0].item(), dataset[-1][0][:, 2][0].item()

    train, tmp = random_split(dataset, [0.6, 0.4], generator=generator)
    val, _ = random_split(tmp, [0.5, 0.5], generator=generator)

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

    in_features = 3 if only_grid else 35
    model = PINN(in_dimensions=in_features, hidden_size=hidden_size).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = NSLoss(physics_coef=physics_coef)

    epochs = 300

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        running_train_losses = defaultdict(list)

        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            inputs[:, :2] = inputs[:, :2] / 299
            inputs[:, 2] = (inputs[:, 2] - MIN_TIME) / MAX_TIME
                
            optimizer.zero_grad()

            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            loss.backward()

            running_train_losses['total'].append(loss.item())
            running_train_losses['data'].append(data_loss.item())
            running_train_losses['physics'].append(physics_loss.item())
            
            optimizer.step()

        train_losses['total'].append(np.mean(running_train_losses['total']))
        train_losses['data'].append(np.mean(running_train_losses['data']))
        train_losses['physics'].append(np.mean(running_train_losses['physics']))

        model.eval()
        running_val_losses = defaultdict(list)

        for inputs, targets in val_loader:
            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            inputs[:, :2] = inputs[:, :2] / 299
            inputs[:, 2] = (inputs[:, 2] - MIN_TIME) / MAX_TIME

            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            running_val_losses['total'].append(loss.item())
            running_val_losses['data'].append(data_loss.item())
            running_val_losses['physics'].append(physics_loss.item())

        val_losses['total'].append(np.mean(running_val_losses['total']))
        val_losses['data'].append(np.mean(running_val_losses['data']))
        val_losses['physics'].append(np.mean(running_val_losses['physics']))

        print(f"Epoch: {epoch} - Train Losses: {train_losses['total'][-1]:.4f}, {train_losses['physics'][-1]*physics_coef:.4f} - \
            Val Losses: {val_losses['total'][-1]:.4f}, {val_losses['physics'][-1]*physics_coef:.4f}")
        

        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"{model_save_path}/model{epoch}.pt")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your PINN model.")

    parser.add_argument("--physics", type=int, default=100, help="Physics loss multiplier")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--only-grid", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-wake", type=bool, action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    model_save_path = f"models/SiLU{args.hidden_size}-{'wake' if args.use_wake else 'no-wake'}"
    os.makedirs(model_save_path, exist_ok=True)

    main(args.physics, args.hidden_size, args.only_grid, args.use_wake, model_save_path)
