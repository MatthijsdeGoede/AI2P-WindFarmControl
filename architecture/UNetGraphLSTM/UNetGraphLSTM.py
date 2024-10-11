from torch import nn
import torch

from architecture.pignn.pignn import PowerPIGNN
from architecture.windspeedLSTM.custom_unet import CustomUNet, CustomUNetUp, CustomUNetDown
from architecture.windspeedLSTM.windspeedLSTM import WindspeedLSTMHelper

class UNetGraphLSTM(nn.Module):
    def __init__(self, sequence_length, input_size,
                 edge_in_dim: int,
                 node_in_dim: int,
                 global_in_dim: int,
                 edge_hidden_dim: int = 32,
                 node_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 output_dim: int = 90000,
                 n_pign_layers: int = 3,
                 num_nodes: int = 10,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None):
        super(UNetGraphLSTM, self).__init__()

        self.unetDown = CustomUNetDown(sequence_length)
        self.center = UNetGraphLSTMHelper(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                          global_hidden_dim, output_dim, n_pign_layers, residual, input_norm,
                                          pign_mlp_params, reg_mlp_params)
        self.unetUp = CustomUNetUp(sequence_length)


        print(sequence_length, input_size)

    def forward(self, x_windspeeds, x_graphs, nf, ef, gf):
        x1, x2, x3, x4, x5 = self.unetDown(x_windspeeds)
        x5 = self.center(x5, x_graphs, nf, ef, gf)
        x = self.unetUp(x5, x4, x3, x2, x1)
        return x

class UNetGraphLSTMHelper(nn.Module):
    def __init__(self,
                 edge_in_dim: int,
                 node_in_dim: int,
                 global_in_dim: int,
                 edge_hidden_dim: int = 32,
                 node_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 output_dim: int = 90000,
                 n_pign_layers: int = 3,
                 num_nodes: int = 10,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None):
        super(UNetGraphLSTMHelper, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(64, 64)
        self.combination = nn.Linear(444, 64)
        self.gnn = PowerPIGNN(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                        global_hidden_dim, output_dim, n_pign_layers, residual, input_norm,
                                        pign_mlp_params, reg_mlp_params)

    def forward(self, x_windspeeds, x_graphs, nf, ef, gf):
        x = self.flatten(x_windspeeds)
        x, _ = self.lstm(x)
        gnn = self.gnn(x_graphs, nf, ef, gf).reshape(-1, 1, 380).repeat(1, 1024, 1)
        concat = torch.cat((x, gnn), dim=2)
        x = self.combination(concat)
        return x.reshape(-1, 1024, 8, 8)
