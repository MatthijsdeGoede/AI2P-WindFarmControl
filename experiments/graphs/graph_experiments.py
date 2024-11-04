import argparse
import json
import os
import torch

import numpy as np
import torch.nn as nn

from datetime import datetime

from torch.optim import Adam
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, ConcatDataset

from architecture.pignn.pignn import FlowPIGNN
from architecture.pignn.deconv import FCDeConvNet, DeConvNet
from architecture.lstm.wind_speed_lstm import WindSpeedLSTM, WindSpeedLSTMDeConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator()
generator.manual_seed(42)


class GraphDataset(Dataset):
    """
    A dataset class for loading graph data from a specified directory.

    Attributes:
        root (str): The directory containing graph files.
        num_samples (int): The total number of graph samples in the dataset.
        data (list): Preloaded graph data if `preload` is True.
        graph_paths (list): List of graph file paths if `preload` is False.
        preload (bool): Flag to indicate whether to preload the graph data into memory.
    """

    def __init__(self, root, preload=True, transform=None, pre_transform=None):
        """
        Initializes the GraphDataset with the given root directory and configuration.

        Args:
            root (str): The directory containing the graph files.
            preload (bool): Whether to preload the data into memory (default: True).
            transform (callable, optional): A function/transform to apply to the data.
            pre_transform (callable, optional): A function/transform to apply to the data before any other transformations.
        """
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.num_samples = len([name for name in os.listdir(self.root) if name != "README.md"])
        self.data = None
        self.graph_paths = None
        self.root = root
        self.preload = preload
        self.load_graph_paths()

    def load_graph_paths(self):
        """
        Loads the graph paths or preloads the graph data based on the `preload` attribute.

        If `preload` is True, it loads the graph data into memory; otherwise, it
        stores the paths to the graph files for later loading.
        """
        if self.preload:
            self.data = [torch.load(f"{self.root}/graph_{i}.pt") for i in range(30005, 42000 + 1, 5)]
        else:
            self.graph_paths = [f"{self.root}/graph_{i}.pt" for i in range(30005, 42000 + 1, 5)]

    def len(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def get(self, idx):
        """
        Retrieves the graph data at the specified index.

        Args:
            idx (int): The index of the graph to retrieve.

        Returns:
            graph: The graph data at the specified index.
        """
        if self.preload:
            return self.data[idx]
        return torch.load(self.graph_paths[idx])


class GraphTemporalDataset(Dataset):
    """
    A dataset class for loading temporal graph data sequences from a specified directory.

    Attributes:
        root (str): The directory containing graph files.
        seq_length (int): The length of the graph sequences.
        preload (bool): Flag to indicate whether to preload the graph data into memory.
        data (list): Preloaded graph data if `preload` is True.
    """

    def __init__(self, root, seq_length, preload=True, transform=None, pre_transform=None):
        """
        Initializes the GraphTemporalDataset with the given parameters.

        Args:
            root (str): The directory containing the graph files.
            seq_length (int): The length of the sequences of graphs.
            preload (bool): Whether to preload the data into memory (default: True).
            transform (callable, optional): A function/transform to apply to the data.
            pre_transform (callable, optional): A function/transform to apply to the data before any other transformations.
        """
        super(GraphTemporalDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.seq_length = seq_length
        self.preload = preload

        if preload:
            self.data = [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for start in range(self.len()) for i in range(self.seq_length)]

    def _get_sequence(self, start):
        """
        Retrieves a sequence of graphs starting from the specified index.

        Args:
            start (int): The starting index of the sequence.

        Returns:
            list: A list of graphs forming a sequence.
        """
        if self.preload:
            return [self.data[start + i] for i in range(self.seq_length)]
        else:
            return [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for i in range(self.seq_length)]

    def len(self):
        """
        Returns the number of possible sequences in the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len([name for name in os.listdir(self.root)]) - 2 * self.seq_length

    def get(self, idx):
        """
        Retrieves a sequence of graphs at the specified index along with the next sequence.

        Args:
            idx (int): The index of the starting graph in the sequence.

        Returns:
            tuple: A tuple containing the current sequence of graphs and the next sequence of graphs.
        """
        return self._get_sequence(idx), self._get_sequence(idx + self.seq_length)


def get_dataset(dataset_dirs, is_temporal, seq_length):
    """
    Loads and concatenates datasets from specified directories.

    Args:
        dataset_dirs (list): List of directories containing dataset files.
        is_temporal (bool): Flag to indicate if the datasets are temporal.
        seq_length (int): The length of sequences if datasets are temporal.

    Returns:
        Dataset: A concatenated dataset containing samples from all specified directories.
    """
    datasets = []
    for path in dataset_dirs:
        dataset = GraphTemporalDataset(root=path, seq_length=seq_length) if is_temporal else GraphDataset(root=path)
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    print(f"Loaded datasets, {len(dataset)} samples")
    return dataset


def custom_collate_fn(batch):
    """
    Custom collate function for batching graph sequences.

    Args:
        batch (list): A batch of sequences, each a list of graphs.

    Returns:
        list: A list of batched graph data.
    """
    # Each batch consists of a list of sequences (each a list of graphs)
    return [Data.from_data_list(seq) for seq in batch]


def create_data_loaders(dataset, batch_size, seq_length):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        dataset (Dataset): The dataset to create loaders from.
        batch_size (int): The number of samples per batch.
        seq_length (int): The length of sequences for temporal datasets.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    collate = custom_collate_fn if seq_length > 1 else None
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_loss(batch, criterion, model):
    """
    Computes the loss for a given batch of data using the specified model and loss criterion.

    This function handles different types of models (e.g., fully connected networks vs.
    graph neural networks) by concatenating input features appropriately.

    Args:
        batch (Batch): A batch of data containing features, positions, edge attributes,
                       global features, and targets.
        criterion (callable): A loss function used to compute the loss.
        model (nn.Module): The model used for making predictions.

    Returns:
        torch.Tensor: The computed loss value for the given batch.
    """
    # Logic to handle different model types
    x, pos, edge_attr, glob, target = batch.x, batch.pos, batch.edge_attr.float(), batch.global_feats.float(), batch.y
    # Concatenate features for non-GNN models
    if isinstance(model, FCDeConvNet):
        x_cat = torch.cat([x.flatten(), pos.flatten(), edge_attr.flatten(), glob.flatten()], dim=-1).float()
        pred = model(x_cat)
    else:
        nf = torch.cat((x, pos), dim=-1).float()
        pred = model(batch, nf, edge_attr, glob)
    loss = criterion(pred, target.reshape((pred.size(0), -1)))
    return loss


def train_epoch(train_loader, model, criterion, optimizer, scheduler):
    """
    Trains the model for one epoch using the provided training data loader.

    This function computes the loss for each batch, performs backpropagation,
    updates the model parameters using the optimizer, and adjusts the learning
    rate with the scheduler.

    Args:
        train_loader (DataLoader): The data loader for training data.
        model (nn.Module): The model to be trained.
        criterion (callable): The loss function used to compute the loss.
        optimizer (Optimizer): The optimizer used for updating the model parameters.
        scheduler (Scheduler): The learning rate scheduler to adjust the learning rate.

    Returns:
        float: The average training loss for the epoch.
    """
    train_losses = []
    model.train()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss = compute_loss(batch, criterion, model)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_epoch(val_loader, model, criterion):
    """
    Evaluates the model on the validation dataset for one epoch.

    This function computes the loss for each batch in the validation set
    without updating the model parameters, ensuring that the model remains
    in evaluation mode throughout the process.

    Args:
        val_loader (DataLoader): The data loader for validation data.
        model (nn.Module): The model to be evaluated.
        criterion (callable): The loss function used to compute the loss.

    Returns:
        float: The average validation loss for the epoch.
    """
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            batch = batch.to(device)
            val_loss = compute_loss(batch, criterion, model)
            val_losses.append(val_loss.item())
    model.train()
    return np.mean(val_losses)


def train(model, train_params, train_loader, val_loader, output_folder):
    """
    Trains the given model using the provided training parameters and data loaders.

    This function manages the training and validation process over a specified number of epochs,
    saves the model checkpoints, and implements early stopping based on validation loss.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_params (dict): A dictionary containing training parameters, including:
            - num_epochs (int): The total number of epochs for training.
            - early_stop_after (int): The number of epochs to wait before stopping if there is no improvement.
        train_loader (DataLoader): The data loader for training data.
        val_loader (DataLoader): The data loader for validation data.
        output_folder (str): The folder where model checkpoints and loss data will be saved.

    Returns:
        None: The function saves model checkpoints and training loss data, but does not return any values.
    """
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loss_list = []
    val_loss_list = []

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss = eval_epoch(val_loader, model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")

        # Check early stopping criterion
        if epoch == num_epochs:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{output_folder}/pignn_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
            print(f'Early stopping at epoch {epoch}')
            break


def process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size, output_size):
    """
    Processes a batch of temporal graph sequences to generate predictions and compute the loss.

    This function handles a sequence of graphs for each item in the batch, passing them
    through a graph model to generate embeddings, which are then processed by a temporal model
    to produce the final output. The loss is computed against the target outputs.

    Args:
        batch (tuple): A tuple containing two elements:
            - A list of sequences of graph objects.
            - A list of target outputs corresponding to each sequence.
        graph_model (nn.Module): The model used to process individual graphs and generate embeddings.
        temporal_model (nn.Module): The model used to process the temporal sequence of embeddings.
        criterion (callable): The loss function used to compute the loss between output and target.
        embedding_size (tuple): A tuple specifying the dimensions of the output embeddings from the graph model.
        output_size (tuple): A tuple specifying the dimensions of the expected output from the temporal model.

    Returns:
        torch.Tensor: The computed loss between the model output and the target output.
    """
    generated_img = []
    target_img = []
    for i, seq in enumerate(batch[0]):
        # Process graphs in parallel at each timestep for the entire batch
        seq = seq.to(device)
        nf = torch.cat((seq.x.to(device), seq.pos.to(device)), dim=-1).float()
        ef = seq.edge_attr.to(device).float()
        gf = seq.global_feats.to(device).float()
        graph_output = graph_model(seq, nf, ef, gf).reshape(-1, embedding_size[0], embedding_size[1])
        generated_img.append(graph_output)
        target_img.append(batch[1][i].y.to(device).reshape(-1, output_size[0], output_size[1]))

    temporal_img = torch.stack(generated_img, dim=1)
    output = temporal_model(temporal_img).flatten()
    target = torch.stack(target_img, dim=1).flatten()
    return criterion(output, target)


def train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler, embedding_size,
                         output_size):
    """
    Trains the graph and temporal models for one epoch using the provided training data loader.

    This function processes batches of data through the graph and temporal models, computes
    the loss for each batch, and updates the model parameters using backpropagation.

    Args:
        train_loader (DataLoader): The data loader for training data containing sequences of graphs.
        graph_model (nn.Module): The model used to process individual graphs.
        temporal_model (nn.Module): The model used to process sequences of graph embeddings.
        criterion (callable): The loss function used to compute the loss.
        optimizer (Optimizer): The optimizer used to update model parameters.
        scheduler (Scheduler): The learning rate scheduler to adjust the learning rate.
        embedding_size (tuple): A tuple specifying the dimensions of the output embeddings from the graph model.
        output_size (tuple): A tuple specifying the dimensions of the expected output from the temporal model.

    Returns:
        float: The average training loss for the epoch.
    """
    train_losses = []
    graph_model.train()
    temporal_model.train()
    for i, batch in enumerate(train_loader):
        if i < len(train_loader) - 1:
            loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size, output_size)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return np.mean(train_losses)


def eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size, output_size):
    """
    Evaluates the graph and temporal models on the validation dataset for one epoch.

    This function computes the loss for each batch in the validation set without updating
    the model parameters, ensuring that the models remain in evaluation mode.

    Args:
        val_loader (DataLoader): The data loader for validation data containing sequences of graphs.
        graph_model (nn.Module): The model used to process individual graphs.
        temporal_model (nn.Module): The model used to process sequences of graph embeddings.
        criterion (callable): The loss function used to compute the loss.
        embedding_size (tuple): A tuple specifying the dimensions of the output embeddings from the graph model.
        output_size (tuple): A tuple specifying the dimensions of the expected output from the temporal model.

    Returns:
        float: The average validation loss for the epoch.
    """
    with torch.no_grad():
        graph_model.eval()
        temporal_model.eval()
        val_losses = []
        for i, batch in enumerate(val_loader):
            if i < len(val_loader) - 1:
                val_loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size,
                                                  output_size)
                val_losses.append(val_loss.item())
    graph_model.train()
    temporal_model.train()
    return np.mean(val_losses)


def train_temporal(graph_model, temporal_model, train_params, train_loader, val_loader, output_folder, embedding_size,
                   output_size):
    """
    Trains the temporal model using the provided graph model, training parameters, and data loaders.

    This function coordinates the training process over a specified number of epochs, including
    model training, validation, saving checkpoints, and implementing early stopping based on
    validation loss.

    Args:
        graph_model (nn.Module): The model used to process individual graphs.
        temporal_model (nn.Module): The model used to process sequences of graph embeddings.
        train_params (dict): A dictionary containing training parameters, including:
            - num_epochs (int): The total number of epochs for training.
            - early_stop_after (int): The number of epochs to wait before stopping if there is no improvement.
        train_loader (DataLoader): The data loader for training data containing sequences of graphs.
        val_loader (DataLoader): The data loader for validation data containing sequences of graphs.
        output_folder (str): The folder where model checkpoints and loss data will be saved.
        embedding_size (tuple): A tuple specifying the dimensions of the output embeddings from the graph model.
        output_size (tuple): A tuple specifying the dimensions of the expected output from the temporal model.

    Returns:
        None: The function saves model checkpoints and training loss data but does not return any values.
    """
    optimizer = Adam(list(graph_model.parameters()) + list(temporal_model.parameters()), lr=0.01)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loss_list = []
    val_loss_list = []

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler,
                                          embedding_size, output_size)
        val_loss = eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size, output_size)
        learning_rate = optimizer.param_groups[0]['lr']
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointers
        torch.save(graph_model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")
        torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_{epoch}.pt")

        if epoch == num_epochs:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)

        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(graph_model.state_dict(), f"{output_folder}/pignn_best.pt")
            torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
            print(f'Early stopping at epoch {epoch}')
            break


def create_output_folder(train_config, net_type):
    """
    Creates a timestamped output folder for storing results of the training process.

    The output folder name is constructed based on the current timestamp, case number,
    wake steering configuration, network type, and sequence length.

    Args:
        train_config (dict): A configuration dictionary containing training parameters, including:
            - case_nr (int): The case number for the experiment.
            - wake_steering (bool): Indicates if wake steering is enabled.
            - seq_length (int): The length of the input sequences.
        net_type (str): The type of neural network being used (e.g., "pignn_lstm_deconv").

    Returns:
        str: The path of the created output folder.
    """
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config['case_nr']}_{train_config['wake_steering']}_{net_type}_" \
                    f"{train_config['seq_length']}"
    os.makedirs(output_folder)
    return output_folder


def save_config(output_folder, config):
    """
    Saves the training configuration to a JSON file in the specified output folder.

    Args:
        output_folder (str): The path of the folder where the configuration will be saved.
        config (dict): A dictionary containing the configuration settings.

    Returns:
        None: The function writes the configuration to a file but does not return any value.
    """
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)


def get_pignn_config():
    """
    Retrieves the configuration settings for the PIGNN model.

    This function returns a dictionary containing model hyperparameters such as
    input dimensions, layer configurations, and normalization settings.

    Returns:
        dict: A dictionary with PIGNN model configuration settings.
    """
    return {
        'edge_in_dim': 2,
        'node_in_dim': 3,
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
    }


def get_dataset_dirs(case_nr, wake_steering, max_angle, use_all_data):
    """
    Constructs the dataset directory paths based on the specified parameters.

    This function generates paths to data folders based on the case number, whether
    wake steering is used, the maximum angle, and if all data should be considered.

    Args:
        case_nr (int): The case number for the experiment.
        wake_steering (bool): Indicates if wake steering is enabled.
        max_angle (int): The maximum angle for the data.
        use_all_data (bool): If True, includes all available data cases.

    Returns:
        list: A list of directory paths for the datasets.
    """
    cases = [1, 2, 3] if use_all_data else [case_nr]
    wake_steering_cases = [True, False] if use_all_data else [wake_steering]
    folders = []

    for case in cases:
        for steering in wake_steering_cases:
            post_fix = "LuT2deg_internal" if steering else "BL"
            data_folder = f"../../data/Case_0{case}/graphs/{post_fix}/{max_angle}"
            folders.append(data_folder)

    return folders


def get_config(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=1, batch_size=64,
               output_size=128, direct_lstm=False, num_epochs=300, early_stop_after=10, use_all_data=False):
    """
    Retrieves the complete configuration settings for the experiment, including model and training parameters.

    Args:
        case_nr (int): The case number for the experiment.
        wake_steering (bool): Indicates if wake steering is enabled.
        max_angle (int): The maximum angle for the experiment.
        use_graph (bool): If True, use graph representation.
        seq_length (int): The length of the input sequences.
        batch_size (int): The batch size for training.
        output_size (int): The desired output size.
        direct_lstm (bool): If True, feed the PIGNN output directly to the LSTM.
        num_epochs (int): The total number of epochs for training.
        early_stop_after (int): The number of epochs to wait before stopping if there is no improvement.
        use_all_data (bool): If True, use all available training data.

    Returns:
        tuple: A tuple containing the model configuration and training configuration dictionaries.
    """
    return get_pignn_config(), {
        'case_nr': case_nr,
        'wake_steering': wake_steering,
        'max_angle': max_angle,
        'num_epochs': num_epochs,
        'use_graph': use_graph,
        'early_stop_after': early_stop_after,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'direct_lstm': direct_lstm,
        'output_size': output_size,
        "dataset_dirs": get_dataset_dirs(case_nr, wake_steering, max_angle, use_all_data)
    }


def run(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=1, batch_size=64, direct_lstm=False,
        output_size=128, use_all_data=False):
    """
    Main function to execute the training process with the specified configurations.

    This function orchestrates the entire training workflow, including setting up the configuration,
    creating output directories, loading datasets, initializing models, and starting the training process.
    """
    model_cfg, train_cfg = get_config(case_nr=case_nr, wake_steering=wake_steering, max_angle=max_angle,
                                      use_graph=use_graph, seq_length=seq_length, batch_size=batch_size,
                                      output_size=output_size, direct_lstm=direct_lstm, use_all_data=use_all_data)

    is_direct_lstm = train_cfg['direct_lstm']
    is_temporal = seq_length > 1
    pignn_type = ("pignn_lstm_deconv" if is_direct_lstm else "pignn_unet_lstm") if is_temporal else "pignn_deconv"
    net_type = f"{pignn_type}_{train_cfg['max_angle']}" if train_cfg['use_graph'] else "fcn_deconv"

    # Create the output folder
    output_folder = create_output_folder(train_cfg, net_type)
    save_config(output_folder, train_cfg)

    # Get the dataset and data loaders
    dataset = get_dataset(train_cfg['dataset_dirs'], is_temporal, seq_length)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, train_cfg['batch_size'], seq_length)

    # Initialize the graph model
    out_size = (output_size, output_size)
    deconv_model = DeConvNet(1, [64, 128, 256, 1],
                             output_size=output_size) if not is_temporal or not is_direct_lstm else None
    graph_model = FlowPIGNN(**model_cfg, deconv_model=deconv_model).to(device) if \
        train_cfg['use_graph'] else FCDeConvNet(212, 650, 656, 500).to(device)

    print(graph_model)

    # Optionally initialize the temporal model
    if is_temporal:
        temporal_model = WindSpeedLSTMDeConv(seq_length, [64, 128, 256, 1], output_size).to(
            device) if is_direct_lstm else WindSpeedLSTM(seq_length).to(device)
        embedding_size = (50, 10) if is_direct_lstm else out_size
        train_temporal(graph_model, temporal_model, train_cfg, train_loader, val_loader, output_folder, embedding_size, out_size)
    else:
        train(graph_model, train_cfg, train_loader, val_loader, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with different configurations.')
    parser.add_argument('--case_nr', type=int, default=1, help='Case number to use for the experiment (default: 1)')
    parser.add_argument('--wake_steering', action='store_true', help='Enable wake steering (default: False)')
    parser.add_argument('--max_angle', type=int, default=30, help='Maximum angle for the experiment (default: 30)')
    parser.add_argument('--use_graph', action='store_true', help='Use graph representation (default: False)')
    parser.add_argument('--seq_length', type=int, default=1, help='Sequence length for the experiment (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the experiment (default: 64)')
    parser.add_argument('--direct_lstm', action='store_true', help='Feed the PIGNN output directly to the LSTM (default: False)')
    parser.add_argument('--use_all_data', action='store_true', help='Use all available training data (default: False)')
    args = parser.parse_args()

    run(case_nr=args.case_nr, wake_steering=args.wake_steering, max_angle=args.max_angle, use_graph=args.use_graph, 
        seq_length=args.seq_length, batch_size=args.batch_size, direct_lstm=args.direct_lstm, use_all_data=args.use_all_data)
