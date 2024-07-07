# Residual Channel Attention Network (RCAN)

# Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, “Image Super-Resolution Using 
# Very Deep Residual Channel Attention Networks,” in Computer Vision-ECCV 2018, vol. 
# 11211, V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer 
# International Publishing, 2018, pp. 294-310.
# ----------------------------------------------------------------------------------------

from torch import nn
import math

class Upsampler(nn.Module):
    '''
    Args:
    - scale (int): scale factor Default: 4.
    - n_features (int): number of features. Defualt: 8.
    - kernel_size (int): kernel size. Default: 3.
    - bn (Bool): whether to use batch normalization. Defualt: False.
    - act (str): activation function name. Default: False.
    - bias (bool): bias in the convolutional layer. Default: True.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, scale=4, n_features=8, kernel_size=3, bn=False, act=False, bias=True,\
        pm='zeros'):
        super().__init__()
        modules = []
        if (scale & (scale - 1)) == 0: # 2, 4, 8... 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(in_channels=n_features, out_channels=4 * n_features,\
                    kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias, padding_mode=pm))
                modules.append(nn.PixelShuffle(2))
                if bn == True:      modules.append(nn.BatchNorm2d(num_features=n_features))
                if act == 'relu' :  modules.append(nn.ReLU(inplace=True))
                if act == 'prelu':  modules.append(nn.PReLU(num_parameters=n_features))
        elif scale == 3:
            modules.append(nn.Conv2d(in_channels=n_features, out_channels=9 * n_features,\
                kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias, padding_mode=pm))
            modules.append(nn.PixelShuffle(3))
            if bn == True:      modules.append(nn.BatchNorm2d(num_features=n_features))
            if act == 'relu' :  modules.append(nn.ReLU(inplace=True))
            if act == 'prelu':  modules.append(nn.PReLU(num_parameters=n_features))
        else:
            raise NotImplementedError
        
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class CALayer(nn.Module):
    '''
    Channel Attention (CA) layer.
    '''
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1,\
                padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1,\
                padding=0, bias=True),
            nn.Sigmoid(),
        )
    def forward(self,x):
        w = self.avg_pool(x)
        w = self.conv_du(w)
        out = x * w
        return out

class RCAB(nn.Module):
    '''
    Residual Channel Attention Block (RCAB)
    '''
    def __init__(self, n_features, kernel_size, reduction, bias=True, bn=False, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale

        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(in_channels=n_features, out_channels=n_features,\
                kernel_size=kernel_size,padding=(kernel_size//2), bias=bias))
            if bn == True:
                modules_body.append(nn.BatchNorm2d(num_features=n_features))
            if i == 0:
                modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(CALayer(channel=n_features,reduction=reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self,x):
        res = self.body(x).mul(self.res_scale)
        out = res + x
        return out

class ResidualGroup(nn.Module):
    '''
    Residual Group (GP)
    '''
    def __init__(self,n_features,n_resblocks,kernel_size=3,reduction=16, res_scale=1.0):
        super().__init__()
        modules_body = [RCAB(n_features=n_features,kernel_size=kernel_size,reduction=reduction,\
            bias=True,bn=False,res_scale=res_scale) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(in_channels=n_features,out_channels=n_features,\
            kernel_size=kernel_size,padding=(kernel_size//2),bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self,x):
        res = self.body(x)
        out = res + x
        return out

class RCAN(nn.Module):
    '''
    Residaul Channel Attention Network (RCAN)
    '''
    def __init__(self, scale=4, n_colors=3, n_resgroups=10, n_resblocks=20, n_features=64, kernel_size=3,\
        reduction=16, res_scale=1.0):
        super().__init__()
        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels=n_colors, out_channels=n_features, kernel_size=kernel_size,\
            padding=(kernel_size // 2), stride=1, bias=True)
        
        # Residual in residaul (RIR) deep feature extraction
        modules_body = [ResidualGroup(n_features=n_features, kernel_size=kernel_size, reduction=reduction,\
            res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size,\
            padding=(kernel_size // 2), stride=1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # Upscale and reconstruction
        self.tail = nn.Sequential(
            Upsampler(scale=scale, n_features=n_features, kernel_size=kernel_size, bn=False, act=False,\
                bias=True),
            nn.Conv2d(in_channels=n_features, out_channels=n_colors, kernel_size=kernel_size,\
                padding=(kernel_size // 2), stride=1, bias=True),
        )

    def forward(self,x):
        x = self.head(x)

        res = self.body(x)
        res = res + x

        out = self.tail(res)
        return out
