from torch import nn
import torch

from architecture.windspeedLSTM.custom_unet import CustomUNet

torch.set_default_dtype(torch.float64)

class WindspeedLSTM(nn.Module):
    def __init__(self, sequence_length, input_size):
        super(WindspeedLSTM, self).__init__()
        center_nn = WindspeedLSTMHelper()
        self.unet = CustomUNet(sequence_length, sequence_length, center_nn)
        print(sequence_length, input_size)

    def forward(self, x):
        x = self.unet(x)
        return x


class WindspeedLSTMHelper(nn.Module):
    def __init__(self):
        super(WindspeedLSTMHelper, self).__init__()

        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(64, 64, dtype=torch.float64)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = self.lstm(x)
        return x.reshape(-1, 1024, 8, 8)