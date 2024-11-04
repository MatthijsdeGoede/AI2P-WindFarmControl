""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A block that consists of two convolutional layers, each followed by batch normalization
    and ReLU activation. This block is typically used to capture features at various levels
    of abstraction.

    Attributes:
        double_conv (nn.Sequential): A sequential container for the convolutional operations.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initializes the DoubleConv block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels in the middle convolution.
                                           If None, it defaults to out_channels.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W) where N is the batch size,
                             C_in is the number of input channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after applying two convolutional layers,
                          batch normalization, and ReLU activation.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    A downsampling block that first applies max pooling followed by a DoubleConv block.
    This is used to reduce the spatial dimensions of the input while capturing features.

    Attributes:
        maxpool_conv (nn.Sequential): A sequential container for max pooling and DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the Down block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward pass through the Down block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after downsampling and applying DoubleConv.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    An upsampling block that performs either bilinear upsampling or transposed convolution
    followed by a DoubleConv block. It combines features from the corresponding downsampling block.

    Attributes:
        up (nn.Module): Upsampling method (bilinear or transposed convolution).
        conv (DoubleConv): DoubleConv block for further processing.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Initializes the Up block.

        Args:
            in_channels (int): Number of input channels from the previous layer.
            out_channels (int): Number of output channels for the next layer.
            bilinear (bool, optional): If True, uses bilinear upsampling;
                                        otherwise uses transposed convolution (default: True).
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the Up block.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer.
            x2 (torch.Tensor): Corresponding feature map from the downsampling path.

        Returns:
            torch.Tensor: Output tensor after upsampling, padding, and applying DoubleConv.
        """
        x1 = self.up(x1)
        # Calculate padding differences to match the dimensions of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Concatenate along the channel dimension
        return self.conv(x)


class OutConv(nn.Module):
    """
    A block that applies a 1x1 convolution to produce the final output of the U-Net model.
    This layer maps the features from the last convolutional block to the desired output channels.

    Attributes:
        conv (nn.Conv2d): 1x1 convolution layer for output generation.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the OutConv block.

        Args:
            in_channels (int): Number of input channels from the last layer.
            out_channels (int): Number of output channels for the final output.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the OutConv block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying 1x1 convolution.
        """
        return self.conv(x)
