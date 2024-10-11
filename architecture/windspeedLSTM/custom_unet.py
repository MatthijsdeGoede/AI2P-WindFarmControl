""" Full assembly of the parts to form the complete network """
from six import print_

from architecture.gch_unet.unet_parts import *
from utils.timing import *


class CustomUNet(nn.Module):
    def __init__(self, n_channels, n_classes, center_nn, bilinear=False):
        super(CustomUNet, self).__init__()
        self.down = CustomUNetDown(n_channels, bilinear)
        self.center = center_nn
        self.up = CustomUNetUp(n_classes, bilinear)


    def forward(self, x):
        x1, x2, x3, x4, x5 = self.down(x)
        x5 = self.center(x5)
        x = self.up(x5, x4, x3, x2, x1)
        return x


class CustomUNetDown(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(CustomUNetDown, self).__init__()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class CustomUNetUp(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(CustomUNetUp, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits