import re
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


def load_data(data_type: str, only_grid: bool):
    dataset = GridDataset(dir="data/Case_01/measurements_flow/postProcessing_BL/winSpeedMapVector/",
                        turbine_csv="data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv",
                        wind_csv="data/Case_01/winDir_processed.csv", 
                        data_type=data_type, 
                        wake_dir="data/Case_01/measurements_flow/postProcessing_LuT2deg_internal/winSpeedMapVector/",
                        wake_turbine_csv="data/Case_01/measurements_turbines/30000_LuT2deg_internal/rot_yaw_combined.csv",
                        only_grid_values=only_grid)

    MIN_TIME, MAX_TIME = dataset[0][0][:, 2][0].item(), dataset[-1][0][:, 2][0].item()

    train, tmp = random_split(dataset, [0.6, 0.4], generator=generator)
    val, _ = random_split(tmp, [0.5, 0.5], generator=generator)

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

    return train_loader, val_loader, MIN_TIME, MAX_TIME


def load_model_optimizer(model, optimizer, model_save_path):
    """
    Loads the latest model of the last training and modifies the model and
    optimizer in place
    """
    saved_models = os.listdir(model_save_path)
    #Extracts the number from file names (e.g., model120.pt to 120)
    #Then, we get the max to find the latest saved epoch
    max_epoch = max([int(re.search(r'\d+', file).group()) for file in saved_models])

    latest_model_save = os.path.join(model_save_path, f"model{max_epoch}.pt")
    checkpoint = torch.load(latest_model_save)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return max_epoch, checkpoint["train_losses"], checkpoint["val_losses"]


def main(physics_coef: int, 
         hidden_size: int, 
         tanh: bool, 
         only_grid: bool, 
         data_type: str, 
         model_save_path: str, 
         use_checkpoint: bool, 
         increase_physics: bool,
         use_sampling: bool,
         sample_size: int
    ):

    train_loader, val_loader, MIN_TIME, MAX_TIME = load_data(data_type, only_grid)

    in_features = 3 if only_grid else 35

    model = PINN(in_dimensions=in_features, hidden_size=hidden_size, tanh_activation=tanh).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = NSLoss(physics_coef=physics_coef, target_epoch=150, max_value=1000)

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    start_epoch = 0
    end_epoch = 300
    if (use_checkpoint):
        start_epoch, train_losses, val_losses = load_model_optimizer(model, optimizer, model_save_path)
        end_epoch = start_epoch + 300

        if (increase_physics): criterion.set_physics_on_epoch(start_epoch)

    for epoch in tqdm(range(start_epoch+1, end_epoch+1)):
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

        print(f"Epoch: {epoch} - Train Losses: {train_losses['total'][-1]:.4f}, {train_losses['physics'][-1]*criterion.physics_coef:.4f} - \
            Val Losses: {val_losses['total'][-1]:.4f}, {val_losses['physics'][-1]*criterion.physics_coef:.4f}")
        
        if (increase_physics): criterion.increase()

        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"{model_save_path}/model{epoch}.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your PINN model.")

    parser.add_argument("--physics", type=int, default=10, help="Physics loss multiplier")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--tanh", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--only-grid", type=bool, action=argparse.BooleanOptionalAction, default=False)
    #possible options are: wake, no-wake, both
    parser.add_argument("--data-type", type=str, default="both")
    parser.add_argument("--use-checkpoint", type=bool, default=False, action=argparse.BooleanOptionalAction)
    #Allowing this option overrides --physics argument
    parser.add_argument("--increase-physics", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--sample", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample-size", type=int, default=256)

    args = parser.parse_args()

    physics = args.physics
    model_type = "Tanh" if args.tanh else "SiLU"
    model_save_path = f"models/{model_type}{args.hidden_size}-p{args.physics}-{f's{args.sample_size}' if args.sample else 'f'}-{args.data_type}"

    if (args.increase_physics):
        physics = 1
        model_save_path = f"models/{model_type}{args.hidden_size}-pVar-{f's{args.sample_size}' if args.sample else 'f'}-{args.data_type}"

    if (args.save_path):
        model_save_path = args.save_path

    os.makedirs(model_save_path, exist_ok=True)

    main(physics, 
        args.hidden_size, 
        args.tanh, 
        args.only_grid, 
        args.data_type, 
        model_save_path, 
        args.use_checkpoint, 
        args.increase_physics,
        args.sample,
        args.sample_size
    )
