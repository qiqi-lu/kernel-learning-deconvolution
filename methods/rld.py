import torch
import tqdm

class RLD(object):
    '''
    Args:
    - kernel (array): convolutional kernel, (out_channels, in_channels/groups, kH, kW)
    - data_range (tuple): (min, max). Default: (0.0, 1.0).
    '''
    def __init__(self, kernel, kernel_bp=None, scale_factor=1, data_range=(0.0, 1.0), pbar_disable=False, device=None):
        # kernel = torch.tensor(kernel, dtype=torch.float32)
        if len(kernel.shape) == 2: kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)

        self.kernel = kernel
        if kernel_bp == None:
            self.kernel_hat = torch.flip(kernel, dims=[-1, -2])
        else:
            self.kernel_hat = kernel_bp
        self.kernel_size = kernel.shape[-1]
        self.pad = self.kernel_size // 2
        self.data_range = data_range
        self.eps = 0.000001
        self.pbar_disbale = pbar_disable
        self.scale_factor = scale_factor

        self.groups = 1
        self.device = device
    
    def to_tensor(self, x):
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 2: x = x.unsqueeze(dim=0).unsqueeze(dim=0)
        if len(x.shape) == 3: x = x.unsqueeze(dim=0).transpose(dim0=-1, dim1=1)
        self.groups = x.shape[1]
        return x

    def conv(self, x):
        # forward project
        x = self.to_tensor(x)
        x = torch.nn.functional.pad(input=x, pad=(self.pad,)*4, mode='reflect')
        x_blur = torch.nn.functional.conv2d(input=x, weight=self.kernel.repeat((self.groups, 1, 1, 1)), groups=self.groups)
        return x_blur

    def decov(self, x, num_iter=100):
        '''Deconvolution
        '''
        x = self.to_tensor(x)

        # initialization
        est = x
        est = torch.nn.functional.interpolate(est, scale_factor=self.scale_factor, mode='nearest-exact')
        est = torch.clamp(est, min=self.data_range[0], max=self.data_range[1])

        pbar = tqdm.tqdm(total=num_iter, desc='RLD', leave=True, ncols=100, disable=self.pbar_disbale)
        
        for _ in range(num_iter):
            est = torch.where(est == 0, self.eps, est)
            # forward projection
            est_conv = torch.nn.functional.pad(input=est, pad=(self.pad,)*4, mode='reflect')
            est_conv = torch.nn.functional.conv2d(input=est_conv, weight=self.kernel.repeat((self.groups, 1, 1, 1)), groups=self.groups)
            est_conv = torch.nn.functional.avg_pool2d(input=est_conv, kernel_size=self.scale_factor, stride=self.scale_factor)

            # divide
            dv = torch.where(est_conv < self.eps, 0, x / est_conv)

            # backward projecion
            dv = torch.nn.functional.interpolate(input=dv, scale_factor=self.scale_factor, mode='nearest-exact')
            dv = torch.nn.functional.pad(input=dv, pad=(self.pad,)*4, mode='reflect')
            update = torch.nn.functional.conv2d(input=dv, weight=self.kernel_hat.repeat((self.groups, 1, 1, 1)), groups=self.groups)

            # update
            est = est * update
            est = torch.clamp(est, min=self.data_range[0], max=self.data_range[1])
            pbar.update(1)
        pbar.close()
        return est
    
    def backward_proj(self, x):
        # backward projecion
        x = self.to_tensor(x)
        dv = torch.nn.functional.pad(input=x, pad=(self.pad,)*4, mode='reflect')
        update = torch.nn.functional.conv2d(input=dv, weight=self.kernel_hat.repeat((self.groups, 1, 1, 1)), groups=self.groups)
        return update
    