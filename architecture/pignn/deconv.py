from torch import nn, relu

from architecture.pignn.mlp import MLP


class DeConvNet(nn.Module):
    def __init__(self, input_channels, layer_channels, output_size=(128, 128)):
        super(DeConvNet, self).__init__()
        # Upsampling layers, each layer multiplies both dims by 2
        layers = []
        for i, out_channels in enumerate(layer_channels):
            in_channels = input_channels if i == 0 else layer_channels[i-1]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())

        self.de_conv = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x):
        output_tensor = self.de_conv(x)
        return output_tensor.flatten(start_dim=1)


class FCDeConvNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(FCDeConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.mlp = MLP(input_dim=500, output_dim=64, num_neurons=[128, 128, 64], hidden_act='ReLU')
        self.de_conv = DeConvNet(1, [64, 128, 256, 1], output_size=output_size)

    def forward(self, x):
        x = x.reshape(-1, 212)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        x = self.mlp(x.reshape(-1, 1, 500))
        return self.de_conv(x.reshape(-1, 1, 8, 8))
