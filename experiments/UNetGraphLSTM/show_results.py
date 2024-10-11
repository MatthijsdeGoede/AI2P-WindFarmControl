import os

import numpy as np
import torch
import json

from architecture.UNetGraphLSTM.UNetGraphLSTM import UNetGraphLSTM
from experiments.UNetGraphLSTM.UnetGrapLSTMDataset import create_data_loaders, UnetGraphLSTMDataset
from experiments.graphs.graph_experiments import get_config
from utils.preprocessing import resize_windspeed
from utils.visualization import plot_prediction_vs_real, animate_prediction_vs_real

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

def create_transform(scale):
    def resize_scalars(windspeed_scalars):
        return [resize_windspeed(scalar, scale) for scalar in windspeed_scalars]
    return resize_scalars


def make_model_predictions(model, inputs, idx, length, dataset):
    graphs, _, _ = dataset.dataset[idx]
    if length < 0:
        graphs = graphs.to(device)
        nf = torch.cat((graphs.x.to(device), graphs.pos.to(device)), dim=-1)
        ef = graphs.edge_attr.to(device)
        gf = graphs.global_feats.to(device)
        return model(inputs, graphs, nf, ef, gf)
    graphs, windspeeds = graphs.to(device), inputs.to(device)
    nf = torch.cat((graphs.x.to(device), graphs.pos.to(device)), dim=-1)
    ef = graphs.edge_attr.to(device)
    gf = graphs.global_feats.to(device)
    output = model(windspeeds, graphs, nf, ef, gf)
    next_outputs = make_model_predictions(model, output, idx + output.shape[1], length - output.shape[1], dataset)
    return torch.cat((output.squeeze(), next_outputs.squeeze()), dim=0)


def get_model_targets(dataset, index, length):
    if length <= 0:
        _, _, targets = dataset[index]
        return targets
    _, _, targets = dataset[index]
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
    batch_size = 1
    scale = config["scale"]

    graph_path = f"../../data/Case_0{case}/graphs/BL/90"
    windspeeds_path = root_dir

    train_loader, val_loader, test_loader = create_data_loaders(graph_path, windspeeds_path, sequence_length, batch_size,
                                                                transform_windspeeds=create_transform(scale))
    model = UNetGraphLSTM(sequence_length=sequence_length, input_size=300,
                          **get_config(case_nr=1, wake_steering=False, max_angle=90, use_graph=True).model).to(device)

    max_epoch = max([int(file.split(".")[0]) for file in os.listdir(f'results/{latest}/model')])
    model.load_state_dict(torch.load(f"results/{latest}/model/{max_epoch}.pt"))
    model.eval()

    print(f"results/{latest}/{max_epoch}.pt")

    transform = create_transform(scale)


    dataset = UnetGraphLSTMDataset(graph_path, windspeeds_path, sequence_length,
                                   transform_windspeed=transform, target_transform_windspeed=transform)


    with (torch.no_grad()):
        graph, windspeeds, targets = next(iter(test_loader))
        animation_length = sequence_length
        graphs, windspeeds = graph.to(device), windspeeds.to(device)
        nf = torch.cat((graphs.x.to(device), graphs.pos.to(device)), dim=-1)
        ef = graphs.edge_attr.to(device)
        gf = graphs.global_feats.to(device)
        outputs = model(windspeeds, graphs, nf, ef, gf)

        outputs, targets = outputs.squeeze(), targets.squeeze()

        def animate_callback(i):
            return outputs[i], targets[i]

        animate_prediction_vs_real(animate_callback, animation_length, f"results/{latest}/animations/case-{case}-max_epoch-{max_epoch}")
        for graphs, windspeeds, targets in test_loader:
            graphs, windspeeds, targets = graphs.to(device), windspeeds.to(device), targets.to(device)
            nf = torch.cat((graphs.x.to(device), graphs.pos.to(device)), dim=-1)
            ef = graphs.edge_attr.to(device)
            gf = graphs.global_feats.to(device)
            output = model(windspeeds, graphs, nf, ef, gf)
            plot_prediction_vs_real(output[0, 0, :, :].cpu(), targets[0, 0, :, :].cpu(), case)
            plot_prediction_vs_real(output[0, sequence_length - 1, :, :].cpu(), targets[0, sequence_length - 1, :, :].cpu(), case)

if __name__ == '__main__':
    plot()