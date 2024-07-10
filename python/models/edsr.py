# Enhanced Deep Super-Resolution Network (EDSR)

# B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, “Enhanced Deep Residual Networks 
# for Single Image Super-Resolution,” in 2017 IEEE Conference on Computer Vision 
# and Pattern Recognition Workshops (CVPRW), Honolulu, HI, USA, 2017, pp. 1132–1140.
# @https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py
# ----------------------------------------------------------------------------------------
import torch.nn as nn
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

class ResBlock(nn.Module):
    '''
    Residual block.

    Args:
    - n_features (int): number of features. Defualt: 128.
    - kernel_size (int): kernel size. Default: 3.
    - bias (bool): bias in the convolutional layer. Default: True.
    - bn (Bool): whether to use batch normalization. Defualt: False.
    - res_scale (float): residual scale factor. Default: 1.0.
    '''
    def __init__(self, n_features=128, kernel_size=3, bias=True, bn=False, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale

        modules = []
        for i in range(2):
            modules.append(nn.Conv2d(in_channels=n_features, out_channels=n_features, stride=1,\
                                kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn == True: modules.append(nn.BatchNorm2d(n_features))
            if i == 0: modules.append(nn.ReLU(inplace=True))
        
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale) # Mult
        res += x
        return res

class backbone(nn.Module):
    '''
    Args:
    - n_colors (int): number of input channels. Default: 3.
    - n_features (int): number of features. Defualt: 128.
    - n_resblocks (int): number of residual blocks. Default: 16.
    - kernel_size (int): kernel size. Default: 3.
    - res_scale (float): residual scale factor. Default: 0.1.
    '''
    def __init__(self, n_colors=3, n_features=128, n_resblocks=16, kernel_size=3, res_scale=0.1):
        super().__init__()
        # head module
        self.head = nn.Conv2d(in_channels=n_colors, out_channels=n_features, kernel_size=kernel_size,
                        stride=1, padding=(kernel_size // 2), bias=True)

        # body modules
        modules_body = []
        for _ in range(n_resblocks):
            modules_body.append(ResBlock(n_features=n_features, kernel_size=kernel_size, bias=True,\
                bn=False, res_scale=res_scale))
        modules_body.append(nn.Conv2d(in_channels=n_features, out_channels=n_features,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=True))
            
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        fea = self.head(x)
        res = self.body(fea)
        out = fea + res
        return out

class EDSR(nn.Module):
    '''
    Args:
    - scale (int): scale factor Default: 4.
    - n_colors (int): number of input channels. Default: 3.
    - n_features (int): number of features. Defualt: 256.
    - n_resblocks (int): number of residual blocks. Default: 32.
    - kernel_size (int): kernel size. Default: 3.
    - res_scale (float): residual scale factor. Default: 0.1.
    '''
    def __init__(self, scale=4, n_colors=3, n_resblocks=32, n_features=256, kernel_size=3, res_scale=0.1):
        super().__init__()
        
        # feature extraction
        self.extracter = backbone(n_colors=n_colors, n_features=n_features, n_resblocks=n_resblocks,\
            kernel_size=kernel_size, res_scale=res_scale)

        # tail modules
        modules_tail = []
        modules_tail.append(Upsampler(scale=scale, n_features=n_features, bn=False, act=False, bias=True))
        modules_tail.append(nn.Conv2d(in_channels=n_features, out_channels=n_colors, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size//2), bias=True))
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        fes = self.extracter(x)
        out = self.tail(fes)
        return out

    


