from torch import nn
import torch

from architecture.pignn.deconv import DeConvNet
from architecture.pignn.mlp import MLP
from architecture.windspeedLSTM.custom_unet import CustomUNet


class WindspeedLSTM(nn.Module):
    def __init__(self, sequence_length):
        super(WindspeedLSTM, self).__init__()
        center_nn = WindspeedLSTMHelper()
        self.unet = CustomUNet(sequence_length, sequence_length, center_nn)

    def forward(self, x):
        x = self.unet(x)
        return x


class WindspeedLSTMHelper(nn.Module):
    def __init__(self):
        super(WindspeedLSTMHelper, self).__init__()

        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(64, 64)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = self.lstm(x)
        return x.reshape(-1, 1024, 8, 8)


class WindSpeedLSTMDeConv(nn.Module):
    def __init__(self, seq_length, de_conv_dims, output_size):
        super(WindSpeedLSTMDeConv, self).__init__()
        self.seq_length = seq_length
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(64, 64, dtype=torch.float32)
        self.mlp = MLP(input_dim=500, output_dim=64, num_neurons=[128, 128, 64], hidden_act='ReLU')
        self.de_conv = DeConvNet(1, de_conv_dims, output_size=output_size)

    def forward(self, x):
        x = self.mlp(x.reshape(-1, 1, 500))
        x = self.flatten(x)
        x, _ = self.lstm(x)
        return self.de_conv(x.reshape(-1, 1, 8, 8))
