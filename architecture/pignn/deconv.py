from torch import nn, relu


class DeConvNet(nn.Module):
    def __init__(self):
        super(DeConvNet, self).__init__()
        # Upsampling layers, each layer multiplies both dims by 2
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        output_tensor = self.deconv(x)
        # Use interpolation to reach the exact desired output size of (300, 300)
        return nn.functional.interpolate(
            output_tensor, size=(128, 128), mode='bilinear', align_corners=False).reshape(output_tensor.size(0), -1)


class FCDeConvNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(FCDeConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.deconv = DeConvNet()

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        x = self.deconv(x.reshape(-1, 1, 10, 50))
        return x