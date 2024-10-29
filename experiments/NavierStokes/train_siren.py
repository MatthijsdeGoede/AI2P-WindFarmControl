import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from Siren import HSiren
from GridDataset import GridDataset
from IncNSLoss import NSLoss

assert torch.cuda.is_available()

generator = torch.Generator()
generator.manual_seed(42)

def main(physics_coef: float, hidden_size: int, wh: int, epochs: int, model_save_path: str, writer):
    dataset = GridDataset(dir="data/Case_01/measurements_flow/postProcessing_BL/winSpeedMapVector/",
                        turbine_csv="data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv",
                        wind_csv="data/Case_01/winDir_processed.csv", 
                        data_type="wake",
                        wake_dir="data/Case_01/measurements_flow/postProcessing_LuT2deg_internal/winSpeedMapVector/",
                        wake_turbine_csv="data/Case_01/measurements_turbines/30000_LuT2deg_internal/rot_yaw_combined.csv",
                        only_grid_values=False,
                        sampling=True,
                        samples_per_grid=128,
                        top_vorticity=0.80)

    MIN_TIME, MAX_TIME = dataset[0][0][:, 2][0].item(), dataset[-1][0][:, 2][0].item()
    RANGE_TIME = MAX_TIME - MIN_TIME
    AVG_TIME = (MAX_TIME + MIN_TIME) / 2

    train, val, _ = random_split(dataset, [0.6, 0.2, 0.2])

    train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)

    model = HSiren(in_features=35,
                  out_features=4,
                  hidden_features=hidden_size,
                  hidden_layers=5,
                  outermost_linear=True,
                  hidden_omega_0=wh).cuda()
    
    criterion = NSLoss(physics_coef=physics_coef)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=20,
                                                           factor=0.1,
                                                           threshold=1e-3,
                                                           min_lr=1e-7)

    with torch.no_grad():
        writer.add_graph(model, dataset[-1][0].float().cuda(non_blocking=True))

        nr_layer = 1
        for layer in model.modules():
            if (isinstance(layer, torch.nn.Linear)):
                writer.add_histogram(f"Layer {nr_layer} - Weights", layer.weight.data.flatten(), 0)
                nr_layer += 1

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    for epoch in tqdm(range(1, epochs+1)):
        running_train_losses = defaultdict(list)
        
        for inputs, targets in train_loader:
            model.train()

            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            #normalizing to (-3, 3)
            inputs[:, 2] = (inputs[:, 2] - AVG_TIME) / (RANGE_TIME / 6)
                
            optimizer.zero_grad()
            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            physics_loss.backward(retain_graph=True)
            physics_grad_magnitude = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000000).item()
            optimizer.zero_grad()

            data_loss.backward(retain_graph=True)
            data_grad_magnitude = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000000).item()
            optimizer.zero_grad()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_train_losses['total'].append(loss.item())
            running_train_losses['data'].append(data_loss.item())
            running_train_losses['physics'].append(physics_loss.item())
            running_train_losses['physics_grad'].append(physics_grad_magnitude)
            running_train_losses['data_grad'].append(data_grad_magnitude)
            

        train_losses['total'].append(np.mean(running_train_losses['total']))
        train_losses['data'].append(np.mean(running_train_losses['data']))
        train_losses['physics'].append(np.mean(running_train_losses['physics']))

        writer.add_scalar("Train - Loss", np.mean(running_train_losses['total']), epoch)
        writer.add_scalar("Train - Physics", np.mean(running_train_losses['physics']), epoch)
        writer.add_scalar("Physics Gradient Magnitude", np.mean(running_train_losses['physics_grad']), epoch)
        writer.add_scalar("Data Gradient Magnitude", np.mean(running_train_losses['data_grad']), epoch)

        running_val_losses = defaultdict(list)

        for inputs, targets in val_loader:
            model.eval()

            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            #normalizing to (-3, 3)
            inputs[:, 2] = (inputs[:, 2] - AVG_TIME) / (RANGE_TIME / 6)

            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            running_val_losses['total'].append(loss.item())
            running_val_losses['data'].append(data_loss.item())
            running_val_losses['physics'].append(physics_loss.item())

        epoch_val_loss = np.mean(running_val_losses['total'])
        val_losses['total'].append(epoch_val_loss)
        val_losses['data'].append(np.mean(running_val_losses['data']))
        val_losses['physics'].append(np.mean(running_val_losses['physics']))

        scheduler.step(epoch_val_loss)
        last_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("Validation - Loss", np.mean(running_val_losses['total']), epoch)
        writer.add_scalar("Validation - Physics", np.mean(running_val_losses['physics']), epoch)
        writer.add_scalar("Learning Rate", last_lr, epoch)

        if (epoch % 10 == 0):
            with torch.no_grad():
                nr_layer = 1
                for layer in model.modules():
                    if (isinstance(layer, torch.nn.Linear)):
                        writer.add_histogram(f"Layer {nr_layer} - Weights", layer.weight.data.flatten(), epoch)
                        nr_layer += 1

        # print(f"Epoch: {epoch} - Train Losses: {train_losses['total'][-1]:.4f}, {train_losses['physics'][-1]*physics_coef:.4f} - \
        #     Val Losses: {val_losses['total'][-1]:.4f}, {val_losses['physics'][-1]*physics_coef:.4f}")
        
        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"{model_save_path}/model{epoch}.pt")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train your SIREN model.")
    parser.add_argument("--physics", type=float, default=1, help="Physics loss multiplier")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--wh", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)

    args = parser.parse_args()

    model_save_path = f"models/HSiren{args.wh}-p{args.physics}"
    os.makedirs(model_save_path, exist_ok=True)
    
    writer = SummaryWriter(f"runs/HSiren{args.wh}-p{args.physics}", flush_secs=30)

    try:
        main(args.physics, args.hidden_size, args.wh, args.epochs, model_save_path, writer)
    except:
        pass
    finally:
        writer.close()