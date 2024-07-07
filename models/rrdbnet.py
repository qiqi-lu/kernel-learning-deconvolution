# Residual in Residual Dense Block (RRDB)

# X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and C. C. Loy, “ESRGAN: 
# Enhanced Super-Resolution Generative Adversarial Networks,” in Computer Vision – 
# ECCV 2018 Workshops, vol. 11133, L. Leal-Taixé and S. Roth, Eds. Cham: Springer 
# International Publishing, 2019, pp. 63–79.
# @ https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    '''Dense Block.

    Args:
    - n_features (int): number of features. Default: 64.
    - growth_channel (int): intermediate channels. Default: 32.
    - bias (bool): bias in the convolutional layers. Default: True.
    - res_scale (float): residual scale. Default: 0.2.
    - pm (str): padding mode. Default: 'zeros'.
    '''
    def __init__(self, n_features=64, growth_channels=32, bias=True, res_scale=0.2, pm='zeros', use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=growth_channels,\
            kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=pm)
        if use_bn == True: self.bn1 = nn.BatchNorm2d(num_features=growth_channels)

        self.conv2 = nn.Conv2d(in_channels=n_features + growth_channels,   out_channels=growth_channels,\
            kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=pm)
        if use_bn == True: self.bn2 = nn.BatchNorm2d(num_features=growth_channels)

        self.conv3 = nn.Conv2d(in_channels=n_features + growth_channels*2, out_channels=growth_channels,\
            kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=pm)
        if use_bn == True: self.bn3 = nn.BatchNorm2d(num_features=growth_channels)

        self.conv4 = nn.Conv2d(in_channels=n_features + growth_channels*3, out_channels=growth_channels,\
            kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=pm)
        if use_bn == True: self.bn4 = nn.BatchNorm2d(num_features=growth_channels)

        self.conv5 = nn.Conv2d(in_channels=n_features + growth_channels*4, out_channels=n_features,\
            kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=pm)
        if use_bn == True: self.bn5 = nn.BatchNorm2d(num_features=n_features)
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.res_scale = res_scale
    
    def forward(self, x):
        if self.use_bn == False:
            x1 = self.act(self.conv1(x))
            x2 = self.act(self.conv2(torch.cat((x, x1), dim=1)))
            x3 = self.act(self.conv3(torch.cat((x, x1, x2), dim=1)))
            x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))

        if self.use_bn == True:
            x1 = self.act(self.bn1(self.conv1(x)))
            x2 = self.act(self.bn2(self.conv2(torch.cat((x, x1), dim=1))))
            x3 = self.act(self.bn3(self.conv3(torch.cat((x, x1, x2), dim=1))))
            x4 = self.act(self.bn4(self.conv4(torch.cat((x, x1, x2, x3), dim=1))))
            x5 = self.bn5(self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1)))

        out = x + x5 * self.res_scale
        return out

class RRDB(nn.Module):
    '''
    Residual in Redsidual Dense Block 
    (may be different from that shown in the figure of the original paper, but consistent with
    the provided code in the paper.)

    Args:
    - n_features (int): number of features. Default: 64.
    - growth_channels (int): intermediate channels. Default: 32.
    - res_scale (float): residual scale. Default: 0.2.
    - bias (bool): bias in the convolutional layers. Default: bool.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, n_features=64, growth_channels=32, res_scale=0.2, bias=True, pm='zeros',\
        use_bn=False):
        super().__init__()
        self.DB1 = DenseBlock(n_features=n_features, growth_channels=growth_channels ,bias=bias,\
            res_scale=res_scale, pm=pm, use_bn=use_bn)
        self.DB2 = DenseBlock(n_features=n_features, growth_channels=growth_channels, bias=bias,\
            res_scale=res_scale, pm=pm, use_bn=use_bn)
        self.DB3 = DenseBlock(n_features=n_features, growth_channels=growth_channels, bias=bias,\
            res_scale=res_scale, pm=pm, use_bn=use_bn)

        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.DB1(x)
        res = self.DB2(res)
        res = self.DB3(res)
        out = x + res * self.res_scale
        return out

class backbone(nn.Module):
    '''
    Args:
    - in_channels (int): number of input channels. Default: 3.
    - n_features (int): number of features. Default: 64.
    - n_blocks (int): number of blocks. Default: 23.
    - growth_channels (int): intermediate channels. Default: 32.
    - bias (bool): bias in the convolutional layers. Default: bool.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, in_channels=3, n_features=64, n_blocks=23, growth_channels=32, bias=True, pm='zeros',\
        use_bn=False):
        super().__init__()

        self.head = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=3, stride=1,\
            padding=1, bias=bias, padding_mode=pm)
        
        body_modules = []
        for _ in range(n_blocks):
            body_modules.append(RRDB(n_features=n_features, growth_channels=growth_channels, res_scale=0.2,\
                bias=bias, pm=pm, use_bn=use_bn))
        body_modules.append(
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1,\
                      bias=bias, padding_mode=pm),
        )
        if use_bn == True: 
            body_modules.append(nn.BatchNorm2d(num_features=n_features))
        self.body = nn.Sequential(*body_modules)

    def forward(self, x):
        fea = self.head(x)
        res = self.body(fea)
        fea = fea + res
        return fea

class Upsampler(nn.Module):
    '''
    Upsampler.

    Args:
    - scale_factor (int): scale_factor. Default: 4.
    - n_features (int): number of features. Default: 64.
    - bias (bool): bias in the convolutional layers. Default: bool.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, scale_factor=4, n_features=64, bias=True, pm='zeros'):
        super().__init__()

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)
        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)
        self.conv3 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale_factor == 4:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x,   scale_factor=2, mode='nearest')))
            fea = self.act(self.conv2(nn.functional.interpolate(input=fea, scale_factor=2, mode='nearest')))

        if self.scale_factor == 3:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x, scale_factor=3, mode='nearest')))
            fea = self.act(self.conv2(fea))

        if self.scale_factor == 2:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x, scale_factor=2, mode='nearest')))
            fea = self.act(self.conv2(fea))

        out = self.act(self.conv3(fea))
        return out

class RRDBNet(nn.Module):
    '''
    Args:
    - scale_factor (int): scale_factor. Default: 4.
    - in_channels (int): number of input channels. Default: 3.
    - out_channels (int): number of output channels. Default: 3.
    - n_features (int): number of features. Default: 64.
    - n_blocks (int): number of blocks. Default: 23.
    - growth_channels (int): intermediate channels. Default: 32.
    - bias (bool): bias in the convolutional layers. Default: bool.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, scale_factor=4, in_channels=3, out_channels=3, n_features=64, n_blocks=23, growth_channels=32,\
        bias=True, pm='zeros'):
        super().__init__()
        # feature extraction
        self.extracter = backbone(in_channels=in_channels, n_features=n_features, n_blocks=n_blocks,\
            growth_channels=growth_channels, bias=bias, pm=pm)
        
        # upsampling
        self.upsampler = nn.Sequential(
            Upsampler(scale_factor=scale_factor, n_features=n_features, bias=True, pm=pm),
            nn.Conv2d(in_channels=n_features, out_channels=out_channels, kernel_size=3, stride=1,\
                padding=3 // 2, bias=bias, padding_mode=pm)
        )

    def forward(self,x):
        fea = self.extracter(x)
        out = self.upsampler(fea)
        return out
