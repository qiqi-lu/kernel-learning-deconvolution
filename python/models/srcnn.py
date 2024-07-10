from torch import nn

class backbone(nn.Module):
    def __init__(self, num_channels=3, n_features=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, n_features, kernel_size=5, padding=5 // 2)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class SRCNN(nn.Module):
    '''
    Args:
    - num_channels: number of input channels. Defualt: 1.
    '''
    def __init__(self, in_channels=1, out_channels=1, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.scale_factor > 1:
            x = nn.functional.interpolate(input=x, scale_factor=self.scale_factor, mode='bicubic')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x