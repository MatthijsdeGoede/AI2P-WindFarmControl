import re
import os
import shutil
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from PINN import PINN
from GridDataset3D import GridDataset
from NDNSLoss import NSLoss


assert torch.cuda.is_available()

generator = torch.Generator()
generator.manual_seed(42)

#Choosen 1/9.6 because after non dimensionalization time has a 
#range of (-9.6, 9.6). This makes it (-1, 1).
TIME_SCALING_FACTOR = 1/9.6

def load_data(use_sampling: bool):
    dataset = GridDataset(dir="data/Case_01/measurements_flow/postProcessing_BL/winSpeedMapVector/",
                        turbine_csv="data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv",
                        wind_csv="data/Case_01/winDir_processed.csv", 
                        data_type="wake", 
                        wake_dir="data/Case_01/measurements_flow/postProcessing_LuT2deg_internal/winSpeedMapVector/",
                        wake_turbine_csv="data/Case_01/measurements_turbines/30000_LuT2deg_internal/rot_yaw_combined.csv",
                        sampling=use_sampling,
                        samples_per_grid=256,
                        top_vorticity=0.80, 
                        time_scaling_factor=TIME_SCALING_FACTOR)

    train, val, _ = random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    batch_size = 16 if use_sampling else 1
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    return train_loader, val_loader


def load_model_optimizer(model: PINN, optimizer: torch.optim.Optimizer, model_save_path: str):
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


def main(physics_coef: float, 
        hidden_size: int, 
        depth: int,
        relu: bool, 
        use_checkpoint: bool, 
        use_sampling: bool,
        model_save_path: str, 
        writer):

    train_loader, val_loader = load_data(use_sampling)

    model = PINN(in_dimensions=26, hidden_size=hidden_size, nr_hidden_layer=depth, out_dimensions=4, relu_activation=relu).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = NSLoss(physics_coef=physics_coef, time_scaling_factor=TIME_SCALING_FACTOR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=25,
                                                           factor=0.1,
                                                           threshold=1e-3,
                                                           min_lr=1e-5)    

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    start_epoch = 0
    end_epoch = 300
    if (use_checkpoint):
        start_epoch, train_losses, val_losses = load_model_optimizer(model, optimizer, model_save_path)
        end_epoch += start_epoch
    
    with torch.no_grad():
        nr_layer = 1
        for layer in model.modules():
            if (isinstance(layer, torch.nn.Linear)):
                writer.add_histogram(f"Layer {nr_layer} - Weights", layer.weight.data.flatten(), 0)
                nr_layer += 1


    for epoch in tqdm(range(start_epoch+1, end_epoch+1)):
        model.train()
        running_train_losses = defaultdict(list)

        for inputs, targets in train_loader:
            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)
                
            optimizer.zero_grad()
            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            # physics_loss.backward(retain_graph=True)
            # physics_grad_magnitude = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000000).item()
            # optimizer.zero_grad()

            # data_loss.backward(retain_graph=True)
            # data_grad_magnitude = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000000).item()
            # optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_train_losses['total'].append(loss.item())
            running_train_losses['data'].append(data_loss.item())
            running_train_losses['physics'].append(physics_loss.item())
            # running_train_losses['physics_grad'].append(physics_grad_magnitude)
            # running_train_losses['data_grad'].append(data_grad_magnitude)
            
        train_losses['total'].append(np.mean(running_train_losses['total']))
        train_losses['data'].append(np.mean(running_train_losses['data']))
        train_losses['physics'].append(np.mean(running_train_losses['physics']))

        writer.add_scalar("Train - Loss", np.mean(running_train_losses['total']), epoch)
        if (physics_coef > 1e-6): writer.add_scalar("Train - Physics", np.mean(running_train_losses['physics']), epoch)
        # writer.add_scalar("Physics Gradient Magnitude", np.mean(running_train_losses['physics_grad']), epoch)
        # writer.add_scalar("Data Gradient Magnitude", np.mean(running_train_losses['data_grad']), epoch)        

        model.eval()
        running_val_losses = defaultdict(list)

        for inputs, targets in val_loader:
            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

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
        if (physics_coef > 1e-6): writer.add_scalar("Validation - Physics", np.mean(running_val_losses['physics']), epoch)
        writer.add_scalar("Learning Rate", last_lr, epoch)

        if (epoch % 10 == 0):
            with torch.no_grad():
                nr_layer = 1
                for layer in model.modules():
                    if (isinstance(layer, torch.nn.Linear)):
                        writer.add_histogram(f"Layer {nr_layer} - Weights", layer.weight.data.flatten(), epoch)
                        nr_layer += 1

        print(f"Epoch: {epoch} - Train Losses: {train_losses['total'][-1]:.4f}, {train_losses['physics'][-1]*criterion.physics_coef:.4f} - \
            Val Losses: {val_losses['total'][-1]:.4f}, {val_losses['physics'][-1]*criterion.physics_coef:.4f}")
        

        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"{model_save_path}/model{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your PINN model.")
    parser.add_argument("--physics", type=float, default=10, help="Physics loss multiplier")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--relu", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-checkpoint", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--use-sampling", type=bool, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    physics = args.physics
    model_type = "ReLU" if args.relu else "SiLU"
    model_save_path = f"models2/{model_type}{args.hidden_size}-d{args.depth}-p{args.physics}-{'s' if args.use_sampling else 'f'}"

    if (args.save_path):
        model_save_path = args.save_path

    os.makedirs(model_save_path, exist_ok=True)

    run_dir = f"runs/{model_type}{args.hidden_size}-d{args.depth}-p{args.physics}-{'s' if args.use_sampling else 'f'}"
    writer = SummaryWriter(run_dir, flush_secs=30)

    try: 
        main(physics, args.hidden_size, args.depth, args.relu, args.use_checkpoint, args.use_sampling, model_save_path, writer)

    except KeyboardInterrupt:
        delete = input("Delete tensorboard logs? (y/n) ")
        if (delete == 'y'): shutil.rmtree(run_dir)
        
    finally: writer.close()
