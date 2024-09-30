import os

import numpy as np
import torch
import json

import torch.nn as nn

from architecture.windspeedLSTM.windspeedLSTM import WindspeedLSTM
from experiments.futureWindspeedLSTM.WindspeedMapDataset import create_data_loaders, WindspeedMapDataset
from utils.preprocessing import resize_windspeed
from utils.visualization import plot_prediction_vs_real, animate_prediction_vs_real

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transform(scale):
    def resize_scalars(windspeed_scalars):
        return [resize_windspeed(scalar, scale) for scalar in windspeed_scalars]
    return resize_scalars


def make_model_predictions(model, inputs, length):
    if length < 0:
        return model(inputs)
    outputs = model(inputs)
    next_outputs = make_model_predictions(model, outputs, length - outputs.shape[0])
    return torch.cat((outputs, next_outputs), dim=0)


def get_model_targets(dataset, index, length):
    if length <= 0:
        _, targets = dataset[index]
        return targets
    _, targets = dataset[index]
    sequence_length = targets.shape[0]
    next_targets = get_model_targets(dataset, index + sequence_length, length - sequence_length)
    return torch.cat((targets, next_targets), dim=0)


def plot():

    latest = max(os.listdir("results"))

    with open(f"results/{latest}/config.json", "r") as f:
        config = json.load(f)

    case = config["case"]
    root_dir = config["root_dir"]
    sequence_length = config["sequence_length"]
    batch_size = config["batch_size"]
    scale = config["scale"]

    train_loader, val_loader, test_loader = create_data_loaders(root_dir, sequence_length, batch_size,
                                                                transform=create_transform(scale))
    model = WindspeedLSTM(sequence_length, 300).to(device)

    max_epoch = max(os.listdir(f'results/{latest}/model'))
    model.load_state_dict(torch.load(f"results/{latest}/model/{max_epoch}"))
    model.eval()

    print(f"results/{latest}/{max_epoch}")

    dataset = WindspeedMapDataset(root_dir, sequence_length)


    with torch.no_grad():
        # start = np.random.randint(0, len(dataset))
        # inputs, _ = dataset[start]
        # animation_length = 50
        # outputs = make_model_predictions(model, inputs[None, :, :, :], animation_length).squeeze()
        # targets = get_model_targets(dataset, start, animation_length).squeeze()
        #
        # def animate_callback(i):
        #     return outputs[i], targets[i]
        #
        # animate_prediction_vs_real(animate_callback, animation_length, f"results/{latest}")

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            plot_prediction_vs_real(output[0, 0, :, :].cpu(), targets[0, 0, :, :].cpu(), case)

if __name__ == '__main__':
    plot()