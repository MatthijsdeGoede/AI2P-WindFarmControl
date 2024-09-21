from torch import nn


class DeConvModel(nn.Module):
    def __init__(self):
        super(DeConvModel, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),  # Upsample to 20x100
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),  # Upsample to 40x200
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)   # Upsample to 80x400
        )

    def forward(self, x):
        output_tensor = self.deconv(x)
        # Use interpolation to reach the exact desired output size of (300, 300)
        return nn.functional.interpolate(output_tensor, size=(300, 300), mode='bilinear', align_corners=False).reshape(output_tensor.size(0), -1)
