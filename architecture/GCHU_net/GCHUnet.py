from .unet_model import UNet
from .GCH_block import GCH
import torch.nn as nn
import time

class GCHUNet(nn.Module):
    def __init__(self,
                 x_resolution: int,
                 y_resolution: int,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 x_size: float,
                 y_size: float,
                 n_channels,
                 n_classes,
                 height: float,
                 bilinear=False
                 ):
        super(GCHUNet, self).__init__()

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.x_size = x_size
        self.y_size = y_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.height = height
        self.bilinear = bilinear

        self.gch = GCH(self.x_resolution,self.y_resolution,self.x_bounds,self.y_bounds,self.x_size,self.y_size,
                       self.height)

        self.unet = UNet(self.n_channels, self.n_classes, self.bilinear)

    def forward(self,x):
        """
        :param x: A tuple consisting of five input lists.
        they respectively stand for:
                    x_coordinates_turbines,
                    y_coordinates_turbines,
                    wind_directions,
                    wind_speeds,
                    yaw_angles
        :return: The output tensor from the `unet` model.
        """
        x1,x2,x3,x4,x5 = x
        t1 = time.time()
        gch_out = self.gch(x1,x2,x3,x4,x5)
        t2 = time.time()
        print("gch", t2-t1)
        unet_out = self.unet(gch_out)
        return unet_out
