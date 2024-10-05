""" Full assembly of the parts to form the complete network """
from six import print_

from architecture.gch_unet.unet_parts import *
from utils.timing import *


class CustomUNet(nn.Module):
    def __init__(self, n_channels, n_classes, center_nn, bilinear=False):
        super(CustomUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.center = center_nn
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # start_timer()
        x1 = self.inc(x)
        # print_timer("x1")
        x2 = self.down1(x1)
        # print_timer("down1")
        x3 = self.down2(x2)
        # print_timer("down2")
        x4 = self.down3(x3)
        # print_timer("down3")
        x5 = self.down4(x4)
        # print_timer("down4")
        x = self.center(x5)
        # print_timer("center")
        x = self.up1(x, x4)
        # print_timer("up1")
        x = self.up2(x, x3)
        # print_timer("up2")
        x = self.up3(x, x2)
        # print_timer("up3")
        x = self.up4(x, x1)
        # print_timer("up4")
        logits = self.outc(x)
        # print_timer("unet")
        return logits

    def use_checkpointing(self):
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
