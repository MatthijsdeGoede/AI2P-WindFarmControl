from typing import List
import torch.nn as nn


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with customizable hidden layers, activation functions,
    and optional input normalization and dropout. Designed for flexible feature extraction in neural networks.

    Attributes:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        num_neurons (List[int]): Number of neurons in each hidden layer.
        hidden_act (nn.Module): Activation function used for hidden layers.
        out_act (nn.Module): Activation function used for the output layer.
        layers (nn.ModuleList): Sequential list of Linear and activation layers.
        dropout (nn.Dropout, optional): Dropout layer applied after each hidden layer if dropout_prob > 0.
        input_norm (nn.Module, optional): Normalization layer applied to the input.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: List[int] = [64, 32],
                 hidden_act: str = 'LeakyReLU',
                 out_act: str = 'LeakyReLU',
                 input_norm: str = None,
                 dropout_prob: float = 0.0):
        """
        Initializes the MLP with specified layer dimensions, activation functions,
        optional normalization, and dropout probability.

        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.
            num_neurons (List[int], optional): List specifying the number of neurons in each hidden layer.
            hidden_act (str, optional): Activation function for hidden layers (default: 'LeakyReLU').
            out_act (str, optional): Activation function for the output layer (default: 'LeakyReLU').
            input_norm (str, optional): Type of input normalization ('batch' or 'layer'). If None, no normalization.
            dropout_prob (float, optional): Probability for dropout layers (default: 0.0).
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        # Build the MLP layers with linear transformations and activation functions
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

        # Add dropout if specified
        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)

        # Configure input normalization if specified
        if input_norm is not None:
            if input_norm == 'batch':
                self.input_norm = nn.BatchNorm1d(input_dim)
            elif input_norm == 'layer':
                self.input_norm = nn.LayerNorm(input_dim)
            else:
                raise RuntimeError("Unsupported normalization type. Use 'batch' or 'layer'.")

    def forward(self, xs):
        """
        Forward pass of the MLP, optionally applying input normalization and dropout.

        Args:
            xs (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Apply input normalization if specified
        if hasattr(self, 'input_norm'):
            xs = self.input_norm(xs)

        # Pass through each layer, applying dropout after each hidden layer
        for i, layer in enumerate(self.layers):
            if i != 0 and hasattr(self, 'dropout'):
                xs = self.dropout(xs)
            xs = layer(xs)
        return xs

    def __repr__(self):
        """
        Provides a detailed string representation of the MLP architecture, including layer dimensions,
        activation functions, and any input normalization.

        Returns:
            str: A formatted string summarizing the MLP structure.
        """
        msg = "MLP \n"
        if hasattr(self, 'input_norm'):
            msg += "Input Norm : {} \n".format(self.input_norm)
        msg += "Dimensions : {} \n".format([self.input_dim] + self.num_neurons + [self.output_dim])
        msg += "Hidden Act. : {} \n".format(self.hidden_act)
        msg += "Out Act. : {} \n".format(self.out_act)
        return msg
