import torch
import torch.nn as nn
import numpy as np

def gauss_kernel_2d(shape=[3, 3], sigma=1.0):
    '''
    Create 2D Guassian kernel.
    Args:
    - shape (tuple[int]): kernel shape. Default: (3, 3).
    - sigma (float): kernel std. Default: 1.0.
    '''
    x_data, y_data = np.mgrid[-shape[0]//2 + 1:shape[0]//2 + 1, -shape[1]//2 + 1:shape[1]//2 + 1]

    x_data = np.expand_dims(np.expand_dims(x_data, axis=0), axis=0)
    y_data = np.expand_dims(np.expand_dims(y_data, axis=0), axis=0)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)

    g = torch.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    # normalization
    g = g / torch.max(g) # [1,1,3,3]
    return g

def gauss_kernel_2d_multichannel(shape=[4, 3, 3, 3], stddev=[2.0, 0.5, 1.0, 1.5]):
    '''
    Create multi-channel Guassian kernel for initialization of convolutional layer.
    Args:
    - shape (tuple[int]): kernel shape (out_channels, in_inchannels, kernel_size[0], kernel_size[1]).
    - stddev (tuple[int]): kernel std in each channel.
    '''
    kernels = []
    for i in range(shape[0]):
        kernels.append(gauss_kernel_2d(shape=[shape[2], shape[3]], sigma=stddev[i]))
    init = torch.cat(kernels, dim=0)
    init = init.repeat(1, shape[1], 1, 1) # [out_channels, in_channels, ker0, ker1]

    # random weights to increase randomness. 
    minval, maxval = 0.0, 1.0
    rad  = (maxval - minval) * torch.rand(size=[1, shape[1], 1, 1], dtype=torch.float32) + minval
    init = init * rad
    return init

class FP1(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=3, n_features=4, kernel_size=3, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size//2), bias=bias)
        self.bn_act1 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act2 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv3 = nn.Conv2d(in_channels=n_features * 2, out_channels=in_channels * 2, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act3 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels * 2), nn.Softplus())

        self.act0 = nn.Softplus()

        # initialization of convolution filters (out_channels, in channels, kernel_size, kernel_size)
        with torch.no_grad():
            self.conv1.weight.data = gauss_kernel_2d_multichannel(shape=[n_features, in_channels, kernel_size, kernel_size],\
                stddev=np.linspace(0.5, 2.0, n_features))
            self.conv2.weight.data = gauss_kernel_2d_multichannel(shape=[n_features, n_features, kernel_size, kernel_size],\
                stddev=np.linspace(0.5, 2.0, n_features))
            self.conv3.weight.data = gauss_kernel_2d_multichannel(shape=[in_channels * 2, n_features * 2, kernel_size, kernel_size],\
                stddev=np.linspace(0.5, 2.0, in_channels*2))

    def forward(self, x):
        out_1  = self.bn_act1(self.conv1(x))
        out_2  = self.bn_act2(self.conv2(out_1))
        cat_12 = torch.cat([out_1, out_2], dim=-3)
        out_3  = self.bn_act3(self.conv3(cat_12))
        out = out_3 + self.act0(x.repeat(1, 2, 1, 1))
        return out

class FP2(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=3, n_features=4 ,kernel_size=3, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act1 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=in_channels * 2, kernel_size=kernel_size,\
                        stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act2 = nn.Sequential(nn.BatchNorm2d(num_features=in_channels * 2), nn.Softplus())

        self.act0  = nn.Softplus()

        with torch.no_grad():
            self.conv1.weight.data = gauss_kernel_2d_multichannel(shape=[n_features, in_channels, kernel_size, kernel_size],\
                stddev=np.linspace(0.5, 2.0, n_features))
            self.conv2.weight.data = gauss_kernel_2d_multichannel(shape=[in_channels*2, n_features, kernel_size, kernel_size],\
                stddev=np.linspace(0.5, 2.0, in_channels * 2))

    def forward(self,x):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        out = out_2 + self.act0(x.repeat(1, 2, 1, 1))
        return out

class BP1(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 8.
    - kernel_size (int): kernel size. Default: 3.
    - init_w_std (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=3, n_features=8, kernel_size=3, init_w_std=1.0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act1 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act2 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv3 = nn.Conv2d(in_channels=n_features*2, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act3 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        # initialization of the weight of filters
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv3.weight, mean=0.0, std=init_w_std)

    def forward(self,x):
        out_1  = self.bn_act1(self.conv1(x))
        out_2  = self.bn_act2(self.conv2(out_1))
        cat_12 = torch.cat(tensors=[out_2, out_1], dim=1)
        out_3  = self.bn_act3(self.conv3(cat_12))
        return out_3

class BP2(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 8.
    - kernel_size (int): kernel size. Default: 3.
    - init_w_std (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=3, n_features=8, kernel_size=3, init_w_std=1.0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act1 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act2 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        # initialization of the weight of filters
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)

    def forward(self,x):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        return out_2

class BP1up(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 8.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    - init_w_std (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=8, n_features=4, kernel_size=3, init_w_std=1.0, bias=False):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=n_features,\
            kernel_size=2, stride=(2, 2), bias=bias)
        self.bn_act_trans = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.conv = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.bn_act = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        # initialization of the weight of filters
        nn.init.trunc_normal_(tensor=self.conv_trans.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv.weight, mean=0.0, std=init_w_std)

    def forward(self, x):
        out_trans = self.bn_act_trans(self.conv_trans(x))
        out = self.bn_act(self.conv(out_trans))
        return out

class DV(nn.Module):
    '''
    output = a / (b + eps).
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - eps (float): epsilon. Default: 0.0001.
    '''
    def __init__(self, in_channels=3, eps=0.0001):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, a, b):
        dv = self.bn(torch.div(a, b + self.eps))
        return dv

class MUL(nn.Module):
    '''
    output = a * b.
    Args:
    - in_channels (int): channel number of input image.
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.bn_act = nn.Sequential(nn.BatchNorm2d(num_features=in_channels), nn.Softplus())

    def forward(self, a, b):
        mul = self.bn_act(torch.mul(a, b))
        return mul

class Merge(nn.Module):
    '''
    Args:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): Number of features. Default: 8.
    - kernel_size (int): kernel size. Default: 3.
    - init_w_std (float): std used for weight initialization. Default: 1.0.
    - bias (bool): bias in convolutional layer. Default: False.
    '''
    def __init__(self, in_channels=3 ,n_features=8, kernel_size=3, init_w_std=1.0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(in_channels=n_features + in_channels + in_channels, out_channels=n_features,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=bias) 
        self.conv3 = nn.Conv2d(in_channels=n_features + n_features, out_channels=n_features,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=bias) 

        self.bn_act3 = nn.Sequential(nn.BatchNorm2d(num_features=n_features), nn.Softplus())

        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()

        # initialization of the weight of filters
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)        
        nn.init.trunc_normal_(tensor=self.conv3.weight, mean=0.0, std=init_w_std)        
    
    def forward(self, e1, e2):
        e2_conv = self.act1(self.conv1(e2))

        e_cat = torch.cat(tensors=[e2_conv, e2, e1], dim=1)
        e_cat = self.act2(self.conv2(e_cat))

        merge = torch.cat(tensors=[e2_conv, e_cat], dim=1)
        merge = self.bn_act3(self.conv3(merge))
        return merge
        
class RLN(nn.Module):
    '''
    Args:
    - scale (int): upsampling scale factor. Default: 4.
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    '''
    def __init__(self, scale=4, in_channels=3, n_features=4, kernel_size=3):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        # H1
        self.ave_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.FP1   = FP1(in_channels=in_channels, n_features=n_features, kernel_size=kernel_size, bias=False)
        self.DV1   = DV(in_channels=in_channels)
        self.BP1   = BP1(in_channels=in_channels, n_features=8, kernel_size=kernel_size, init_w_std=1.0, bias=False)
        self.BP1up = BP1up(in_channels=8, n_features=n_features, kernel_size=kernel_size, init_w_std=1.0, bias=False)
        self.MUL1  = MUL(in_channels=in_channels)
        # H2
        self.FP2  = FP2(in_channels=in_channels, n_features=n_features, kernel_size=kernel_size, bias=False)
        self.DV2  = DV(in_channels=in_channels)
        self.BP2  = BP1(in_channels=in_channels, n_features=8, kernel_size=kernel_size, init_w_std=1.0, bias=False)
        self.MUL2 = MUL(in_channels=in_channels)
        # H3
        self.Merge = Merge(in_channels=in_channels, n_features=8, kernel_size=kernel_size, init_w_std=1.0, bias=False)
        
        # upsampling
        # if self.scale > 1:
        #     self.upsampler = common.Upsampler(scale=scale,n_features=8,kernel_size=kernel_size,\
        #                         bn=False,act=False,bias=True)
            
        self.conv_last = nn.Conv2d(in_channels=8, out_channels=in_channels, kernel_size=kernel_size, stride=1,\
            padding=(kernel_size // 2), bias=True)
        
    def forward(self, x):
        # bicubic interpolation
        x = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bicubic') 

        # H1
        Iap = self.ave_pool(x)

        fp1 = self.FP1(Iap)
        fp1 = torch.mean(input=fp1, dim=1, keepdim=True)
        fp1 = torch.cat([fp1] * self.in_channels, dim=1)

        dv1 = self.DV1(Iap, fp1)

        bp1 = self.BP1(dv1)
        bp1up = self.BP1up(bp1)
        bp1up = torch.mean(input=bp1up, dim=1, keepdim=True)
        E1 = self.MUL1(x, bp1up)

        # H2
        fp2 = self.FP2(x)
        fp2 = torch.mean(input=fp2, dim=1, keepdim=True)
        fp2 = torch.cat([fp2] * self.in_channels, dim=1)

        dv2 = self.DV2(x, fp2)

        bp2 = self.BP2(dv2)
        bp2 = bp2 + torch.ones_like(input=bp2)
        bp2 = torch.mean(input=bp2, dim=1, keepdim=True)

        E2  = self.MUL2(E1, bp2)

        # H3
        merge = self.Merge(E1, E2)
        
        out = self.conv_last(merge)
        return out

if __name__ == '__main__':
    kernel = gauss_kernel_2d(shape=[3,3],sigma=1.0)
    print(kernel)
    kernels = gauss_kernel_2d_multichannel(shape=[4,3,3,3],stddev=np.linspace(0.5, 2.0, 4))
    print(kernels[0,0])

    x = torch.zeros(size=(4,1,128,128))
    model = RLN(scale=4, in_channels=1, n_features=4, kernel_size=3)
    o = model(x)
    print(o.shape)
