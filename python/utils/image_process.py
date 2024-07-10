import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
import skimage.util as util
import skimage.io as io
import scipy.io as scio
import scipy.ndimage as ndimage
import os
import numpy as np

def gaussian_kernel(n=3, std=1.0):
    '''
    Geenrate 2D gaussian kernel.
    - n (int), kernel size.
    - std (float), standard deviation.
    '''
    x, y = np.mgrid[-(n//2):n//2 + 1, -(n//2):n//2 + 1]
    hg = np.exp(-(x**2 + y**2) / (2 * (std**2)))
    h  = hg / np.sum(hg)
    return h

def SNR_simu(S, Sn):
    '''
    Measure the signal-to-noise ratio of simulated image.
    
    Args:
    - S, ground truth image.
    - Sn, noisy image.
    '''
    N = Sn - S
    SNR_simu = 10 * np.log10(np.std(S) / np.std(N))
    return SNR_simu

def interpolation(x, scale_factor=1, mode='bicubic'):
    '''
    Args:
    - x (array or Tensor), input image with shape [H, W] or [H, W, C]
    - scale_factor (int): multiplier fo spatial size. Default: 1.0.
    - mode (str): algorithm used for upsampling. Default: `bicubic`.
    '''
    # convert to tensor 
    if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2: x = x.unsqueeze(dim=0).unsqueeze(dim=0)
    if len(x.shape) == 3: x = x.unsqueeze(dim=0).transpose(dim0=3, dim1=1) # (1, C, H, W)

    x = nn.functional.interpolate(input=x, scale_factor=scale_factor, mode='bicubic')
    return x

def to_image(x):
    x = x.squeeze()
    if len(x.shape) == 3: x = x.transpose(dim0=0, dim1=-1) # (1,C,H,W) -> (H,W,C)
    x = x.numpy()
    return x

class ImageDegradation2D(object):
    '''
    Image degradaiton using `bicubic model` or `classical model`.

    Args:
    - scale_factor (int): down-sampling factor.
    - kernel (array): convolution kernel. (kernel_size[0], kernel_size[1])
    - mode (str): noise type, `gaussian`, `poisson`, `poisson-gaussian`. Default: 'gaussian'.
    - std (float): noise std [0., 1.].
    - ratio (int): control the Poisson noise level. Default: 1000.
    - down_sample_mode (str): Method used for down-sampling. `ave`, `left-top`, and `bicubic`. Default: 'ave'.

    Inputs:
    - x (float32): (H,W,C), [0., 1.].

    Outputs:
    -  (float32): (H,W,C), [0., 1.].
    '''
    def __init__(self, scale_factor=1, kernel=None, noise_mode='gaussian', std=0.0, ratio=1000,\
        down_sample_mode='ave'):

        self.scale_factor = scale_factor
        self.kernel = torch.tensor(kernel / np.sum(kernel), dtype=torch.float32)[None, None, ...]
        self.pad = int(kernel.shape[0] // 2)
        self.noise_mode = noise_mode
        self.std = std # relative to 1.0
        self.ratio = ratio
        self.down_sample_mode = down_sample_mode
        self.groups = 1
    
    def to_tensor(self, x):
        # first to use, convert the array to torch tensor.
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 2: x = x.unsqueeze(dim=0).unsqueeze(dim=0) # (1, 1, H, W)
        if len(x.shape) == 3: x = x.unsqueeze(dim=0).transpose(dim0=1, dim1=3) # (1, C, H, W)
        self.groups = x.shape[1]
        return x

    def convolution(self, x):
        x = self.to_tensor(x)
        x = torch.nn.functional.pad(input=x, pad=(self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x_blur = torch.nn.functional.conv2d(input=x, weight=self.kernel.repeat(self.groups, 1, 1, 1), groups=self.groups)
        return x_blur
    
    def downsampling(self, x):
        x = self.to_tensor(x)
        if self.down_sample_mode == 'bicubic': 
            x_down = nn.functional.interpolate(input=x, scale_factor=1. / self.scale_factor, mode='bicubic')

        if self.down_sample_mode == 'ave':
            x_down = torch.nn.functional.avg_pool2d(input=x, kernel_size=self.scale_factor)

        if self.down_sample_mode == 'left-top':
            x_down = x[:, :, 0::self.scale_factor, 0::self.scale_factor]
        return x_down

    def add_noise(self, x):
        x = self.to_tensor(x)
        # Guassian noise
        if self.noise_mode == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=self.std, size=x.shape)
            noise = np.minimum(np.maximum(noise, -2.0 * self.std), 2.0 * self.std)
            x = x + noise

        # Poisson noise
        if self.noise_mode == 'poisson':
            x = x * self.ratio
            x = np.random.poisson(lam=x)
            x = x / self.ratio

        # mixed Poisson and Guassian noise @ RLN
        if self.noise_mode == 'poisson-gaussian':
            x = x * self.ratio
            x = np.random.poisson(lam=x)
            e = np.max(x)
            x = x / e
            noise = np.random.normal(loc=0.0, scale=self.std, size=x.shape)
            noise = np.minimum(np.maximum(noise, -2.0 * self.std), 2.0 * self.std)
            x = x + noise
            x = x * e
            x = x / self.ratio

        # without noise
        if self.noise_mode == 'none':
            x = x
        return x
    
    def to_nparray(self, x):
        x = x.squeeze(dim=0).transpose(dim0=0, dim1=2).numpy() # (1,C,H,W) -> (H,W,C)
        return x.astype(np.float32)

    def clip(self, x):
        x = self.to_nparray(x)
        x = np.clip(x, a_min=0.0, a_max=1.0)
        return x

def generate_edge(img):
    '''
    Generate soft-edge.
    @ https://gitlab.com/junchenglee/seanet-pytorch/-/blob/master/generate_edge.m?ref_type=heads
    '''
    # gray image
    if len(img.shape) == 2: 
        img = np.expand_dims(img, axis=-1)
    
    edge = []
    for i in range(img.shape[-1]):
        img_sc = img[..., i] # single chennel image
        u = np.gradient(img_sc, edge_order=1)
        u0 = u[0] / (np.sqrt(1.0 + u[0]**2 + u[1]**2))
        u1 = u[1] / (np.sqrt(1.0 + u[0]**2 + u[1]**2))
        d = np.gradient(u0)[0] + np.gradient(u1)[1] # divergence
        edge.append(d)
    edge = np.transpose(np.stack(edge), axes=(1, 2, 0))
    return edge

if __name__ == '__main__':
    root_path = os.path.join('F:', os.sep, 'Datasets')
    data_path_hr = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939')
    data_path_lr = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939',\
        'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')
    txt_file_hr = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'hr.txt') 
    txt_file_lr = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'lr.txt') 

    # data_path = os.path.join('data','TinyMicro')
    # txt_file_hr = os.path.join(data_path,'test_txt','hr.txt')
    # txt_file_lr = os.path.join(data_path,'test_txt','lr.txt')

    with open(txt_file_hr) as f: file_names_hr = f.read().splitlines()
    with open(txt_file_lr) as f: file_names_lr = f.read().splitlines()

    id_show = 100 #38
    
    img_path_hr = os.path.join(data_path_hr,file_names_hr[id_show])
    img_path_lr = os.path.join(data_path_lr,file_names_lr[id_show])

    print('Read: \n',img_path_hr,'\n',img_path_lr)
    # Read image
    img_hr = io.imread(img_path_hr)
    img_lr = io.imread(img_path_lr)[...,-1]

    img_hr = (img_hr-np.min(img_hr))/(np.max(img_hr)-np.min(img_hr))

    print('HR:', img_hr.shape)
    print('LR:', img_lr.shape)
    # ################################################################################################
    edge_lr = generate_edge(img_lr)
    edge_hr = generate_edge(img_hr)
    print(edge_lr.shape)

    nr, nc = 2, 2
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(2.4 * nr, 2.4 * nc), dpi=600, constrained_layout=True)
    axes[0,0].imshow(img_lr, cmap='gray', vmin=0.0, vmax=0.6)
    axes[0,1].imshow(edge_lr[..., 0], cmap='gray', vmin=-0.05, vmax=0.05)
    axes[1,0].imshow(img_hr, cmap='gray', vmin=0.0, vmax=0.6)
    axes[1,1].imshow(edge_hr[..., 0], cmap='gray', vmin=-0.05, vmax=0.05)
    plt.savefig('edge_map.png')

    # import torch 
    # a = torch.tensor([[1,2,5,3],[1,6,5,3],[1,9,5,3],[1,0,5,3]])
    # print(torch.gradient(a))

    os._exit(0)
    # ################################################################################################
    # degradation
    std = 255.0*0.03
    ker = gaussian_kernel(n=25,std=2.0)
    # degra = ImageDegradation2D(scale_factor=4,kernel=ker,mode='gaussian',std=std,down_type='left-top')
    degra = ImageDegradation2D(scale_factor=4, kernel=ker, noise_mode='gaussian', std=std, down_sample_mode='ave')
    # degra = ImageDegradation2D(scale_factor=4,kernel=ker,mode='poisson',std=std,ratio=1000)
    # degra = ImageDegradation2D(scale_factor=4,kernel=ker,mode='poisson-gaussian',std=std,ratio=1000)

    # classical model
    img_blur = degra.convolution(img_hr)
    img_blur = degra.downsampling(img_blur)
    img_blur = degra.add_noise(img_blur)
    # bicubic model
    degra = ImageDegradation2D(scale_factor=4, kernel=ker, noise_mode='gaussian', std=std, down_sample_mode='bicubic')
    img_blur_bicubic = degra.downsampling(img_hr)
    img_blur_bicubic = degra.add_noise(img_blur_bicubic)

    print(img_hr.shape,'->',img_blur.shape)

    # ################################################################################################
    # show
    fig,axes = plt.subplots(nrows=1,ncols=4,dpi=600,figsize=(1.7*4,1.7*1),constrained_layout=True)
    axes[0].imshow(img_hr),axes[0].set_title('HR')
    axes[1].imshow(img_lr),axes[1].set_title('LR')
    axes[2].imshow(img_blur),axes[2].set_title('Classical ({:>.2f})'.format(std))
    axes[3].imshow(img_blur_bicubic),axes[3].set_title('Bicubic ({:>.2f})'.format(std))
    plt.savefig(os.path.join('outputs','figures','blur'))

    def hist_color(ax,img_c,title='',std=True):
        ax.hist(img_c[...,0].flatten(),bins=255,range=(0.,255.),histtype='step',color='red')
        ax.hist(img_c[...,1].flatten(),bins=255,range=(0.,255.),histtype='step',color='green')
        ax.hist(img_c[...,2].flatten(),bins=255,range=(0.,255.),histtype='step',color='blue')
        if std==True:
            ax.set_title(title+' ({:>.2f}|{:>.2f}|{:>.2f})'.format(np.std(img_c[...,0]),\
                np.std(img_c[...,1]),np.std(img_c[...,2])))
        if std == False:
            ax.set_title(title)

    fig,axes = plt.subplots(nrows=1,ncols=4,dpi=600,figsize=(1.7*7,1.7),constrained_layout=True)
    hist_color(axes[0],img_hr[:128,:128],title='HR')
    hist_color(axes[1],img_lr[:56,:56],title='LR')
    hist_color(axes[2],img_blur[:56,:56],title='Degradation')
    hist_color(axes[3],img_blur_bicubic[:56,:56],title='Degradation')
    plt.savefig(os.path.join('outputs','figures','hist_background'))


    fig,axes = plt.subplots(nrows=1,ncols=4,dpi=600,figsize=(1.7*7,1.7),constrained_layout=True)
    hist_color(axes[0],img_hr,title='HR',std=False)
    hist_color(axes[1],img_lr,title='LR',std=False)
    hist_color(axes[2],img_blur,title='Degradation',std=False)
    hist_color(axes[3],img_blur_bicubic,title='Degradation',std=False)
    plt.savefig(os.path.join('outputs','figures','hist_img'))

    print('end')




