from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm

from Siren import Siren
from GridDataset import GridDataset
from NSLoss import NSLoss

def main():
    dataset = GridDataset("data/preprocessed/case1/")

    MIN_TIME, MAX_TIME = dataset[0][0][:, 2][0].item(), dataset[-1][0][:, 2][0].item()

    train, tmp = random_split(dataset, [0.7, 0.3])
    val, _ = random_split(tmp, [0.5, 0.5])

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

    model = Siren(in_features=3, 
                  out_features=5,
                  hidden_features=64,
                  hidden_layers=5,
                  outermost_linear=True,
                  hidden_omega_0=3).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    physics_coef = 1000
    criterion = NSLoss(physics_coef=physics_coef)

    epochs = 200

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    for epoch in tqdm(range(1, epochs+1)):
        running_train_losses = defaultdict(list)
        
        for inputs, targets in train_loader:
            model.train()

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

        running_val_losses = defaultdict(list)

        for inputs, targets in val_loader:
            model.eval()

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
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"models/model{epoch}.pt")
                


if __name__ == "__main__":
    main()
