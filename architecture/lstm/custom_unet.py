""" Full assembly of the parts to form the complete U-Net based network """

import torch
from architecture.unet.unet_parts import *
from torch import nn


class CustomUNet(nn.Module):
    """
    A custom U-Net architecture for image segmentation tasks, allowing flexibility in
    network structure with bilinear upsampling and an optional central network module.

    Attributes:
        n_channels (int): Number of input channels, typically 1 for grayscale or 3 for RGB.
        n_classes (int): Number of output classes (e.g., 1 for binary segmentation, >1 for multi-class).
        bilinear (bool): Whether to use bilinear upsampling in the decoder instead of transposed convolutions.
        center_nn (nn.Module): A customizable central network module that is applied at the bottleneck.
    """
    def __init__(self, n_channels, n_classes, center_nn, bilinear=False):
        """
        Initializes the CustomUNet instance, creating encoder, bottleneck, and decoder components.

        Args:
            n_channels (int): Number of input channels in the image (e.g., 1 for grayscale, 3 for RGB).
            n_classes (int): Number of output classes for segmentation.
            center_nn (nn.Module): The module to use at the bottleneck of the U-Net.
            bilinear (bool, optional): If True, uses bilinear upsampling in the decoder. Defaults to False.
        """
        super(CustomUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder pathway with Downsampling
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # Adjust final channel factor based on bilinear setting
        self.down4 = Down(512, 1024 // factor)

        # Central network module
        self.center = center_nn

        # Decoder pathway with Upsampling
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Final output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Defines the forward pass of the U-Net through encoding, bottleneck, decoding,
        and output prediction layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, n_classes, height, width).
        """
        # Encoder pathway
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Central network module
        x = self.center(x5)

        # Decoder pathway with skip connections
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output layer
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        Enables gradient checkpointing on the U-Net to reduce memory usage during training.

        This method uses torch.utils.checkpoint to store intermediate activations
        of each layer, trading off computational efficiency for reduced memory usage.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
