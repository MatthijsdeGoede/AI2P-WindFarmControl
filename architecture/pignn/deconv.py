from torch import nn, relu
from architecture.pignn.mlp import MLP


class DeConvNet(nn.Module):
    """
    A deconvolutional network that upscales an input tensor through multiple transposed convolutional
    layers, designed for spatially structured data output (e.g., images).

    Attributes:
        de_conv (nn.Sequential): Sequential container of ConvTranspose2d and ReLU layers to upsample the input.
        output_size (tuple): Final output dimensions (height, width).
    """

    def __init__(self, input_channels, layer_channels, output_size=(128, 128)):
        """
        Initializes the DeConvNet with transposed convolutional layers for upsampling.

        Args:
            input_channels (int): Number of channels in the input tensor.
            layer_channels (list): List of channel sizes for each layer in the deconvolutional network.
            output_size (tuple, optional): Desired output size (height, width). Defaults to (128, 128).
        """
        super(DeConvNet, self).__init__()
        layers = []

        # Define deconvolutional layers
        for i, out_channels in enumerate(layer_channels):
            in_channels = input_channels if i == 0 else layer_channels[i - 1]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())  # Activation after each transposed convolution layer

        self.de_conv = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x):
        """
        Forward pass of the DeConvNet, which upsamples the input tensor through deconvolutional layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Flattened output tensor of shape (batch_size, output_size).
        """
        output_tensor = self.de_conv(x)
        return output_tensor.flatten(start_dim=1)  # Flatten output for further processing


class FCDeConvNet(nn.Module):
    """
    A fully connected network combined with a deconvolutional network, used for tasks that
    start with a fully connected feature extraction phase followed by deconvolutional layers
    to produce spatially structured data (e.g., images).

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        mlp (MLP): MLP for further feature extraction after fully connected layers.
        de_conv (DeConvNet): Deconvolutional network for upscaling the processed features.
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """
        Initializes the FCDeConvNet with fully connected layers, an MLP, and a deconvolutional network.

        Args:
            input_size (int): Dimension of the input to the first fully connected layer.
            hidden1_size (int): Number of neurons in the first fully connected layer.
            hidden2_size (int): Number of neurons in the second fully connected layer.
            output_size (tuple): Desired final output dimensions (height, width).
        """
        super(FCDeConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.mlp = MLP(input_dim=500, output_dim=64, num_neurons=[128, 128, 64], hidden_act='ReLU')
        self.de_conv = DeConvNet(1, [64, 128, 256, 1], output_size=output_size)

    def forward(self, x):
        """
        Forward pass of the FCDeConvNet, which processes the input with fully connected layers,
        applies an MLP for feature extraction, and generates output using the deconvolutional network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Final output tensor with spatial structure, produced by the DeConvNet.
        """
        x = x.reshape(-1, 212)  # Reshape input as needed for fully connected layers
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)

        # Pass through MLP and reshape for DeConvNet
        x = self.mlp(x.reshape(-1, 1, 500))
        return self.de_conv(x.reshape(-1, 1, 8, 8))  # Reshape and apply deconvolution
