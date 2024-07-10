import torch
import torch.nn as nn

class Positive(nn.Module):
    # weight parameterization
    def forward(self, x): 
        return torch.square(x)

class Symmetric(nn.Module):
    def forward(self, x):
        x = x.triu() + x.triu(1).transpose(-1, -2)
        return x

class SymmetricPositive(nn.Module):
    def forward(self, x):
        x = x.triu() + x.triu(1).transpose(-1, -2)
        x = torch.square(x)
        return x

class SpatialIsotropic(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        x = (x + torch.flip(x, dims=[-1])) / 2.0
        x = x + torch.rot90(x, k=1, dims=[-1, -2]) + torch.rot90(x, k=2, dims=[-1, -2]) + torch.rot90(x, k=3, dims=[-1, -2])
        x = x / 4.0
        # cp = x[..., self.kernel_size//2, self.kernel_size//2]
        # cp = torch.nn.functional.pad(input=cp[..., None, None], pad=(self.kernel_size//2,)*4, mode='constant', value=1.0)
        # x = x * cp
        return x

class SpatialIsotropicPositive(nn.Module):
    def forward(self, x):
        x = (x + torch.flip(x, dims=[-1])) / 2.0
        x = x + torch.rot90(x, k=1, dims=[-1, -2]) + torch.rot90(x, k=2, dims=[-1, -2]) + torch.rot90(x, k=3, dims=[-1, -2])
        x = torch.square(x / 4.0)
        return x
