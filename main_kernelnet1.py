'''
Model training.
'''

import torch, os, time, sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.io as io
from fft_conv_pytorch import fft_conv
from utils import dataset_utils
from utils import evaluation as eva
from models import kernelnet

# ------------------------------------------------------------------------------
print('='*98)
if sys.platform == 'win32': 
    device, num_workers = torch.device("cpu"), 0
    root_path = os.path.join('F:', os.sep, 'Datasets')

if sys.platform == 'linux' or sys.platform == 'linux2': 
    device, num_workers = torch.device("cpu"), 0
    # device, num_workers = torch.device("cuda"), 6
    root_path = 'data'

print('>> Device:', device, 'Num of workers:', num_workers)

# ------------------------------------------------------------------------------
torch.manual_seed(7)          
input_normalization = 0
path_checkpoint     = 'checkpoints'
validation_enable   = False
data_range          = None

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
# dataset_name, dataset_dim = 'tinymicro_synth', 2
# dataset_name, dataset_dim = 'tinymicro_real', 2
# dataset_name, dataset_dim = 'lung3_synth', 2
# dataset_name, dataset_dim = 'CCPs', 2
# dataset_name, dataset_dim = 'F-actin', 2
# dataset_name, dataset_dim = 'msi_synth', 2
# ------------------------------------------------------------------------------
# dataset_name, dataset_dim = 'F-actin_Nonlinear', 2
# dataset_name, dataset_dim = 'Microtubules2', 2
# dataset_name, dataset_dim = 'SimuBeads3D_128', 3
# dataset_name, dataset_dim = 'SimuMix3D_128', 3
# dataset_name, dataset_dim = 'SimuMix3D_256', 3
dataset_name, dataset_dim = 'SimuMix3D_382', 3
# dataset_name, dataset_dim = 'Microtubule', 3
# dataset_name, dataset_dim = 'Microtubule2', 3
# dataset_name, dataset_dim = 'Nuclear_Pore_complex', 3
# dataset_name, dataset_dim = 'Nuclear_Pore_complex2', 3
# dataset_name, dataset_dim = 'ZeroShotDeconvNet', 3

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
if dataset_name in ['tinymicro_synth', 'tinymicro_real']: epochs = 3
if dataset_name in ['lung3_synth']: epochs = 100  # 36, 40, 55

if dataset_name in ['F-actin_Nonlinear', 'Microtubules2']:
    # ker_size_fp, ker_size_bp = 101, 101
    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = (ker_size_fp,)*2
    kernel_size_bp = (ker_size_bp,)*2

    id_range   = [0, 1]
    training_data_size = id_range[1] - id_range[0]
    batch_size = training_data_size
    epochs = 5000
    std_gauss, poisson, ratio, scale_factor =  9, 1, 1, 1

if dataset_name in ['SimuBeads3D_128', 'SimuMix3D_128', 'SimuMix3D_256']:
    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = [ker_size_fp, 31, 31]
    kernel_size_bp = [ker_size_bp, 25, 25]

    id_range = [0, 1]
    training_data_size = id_range[1] - id_range[0]
    batch_size = training_data_size
    epochs = 10000

    std_gauss, poisson, ratio =  0, 0, 1
    # std_gauss, poisson, ratio =  0.5, 1, 1
    # std_gauss, poisson, ratio =  0.5, 1, 0.3
    # std_gauss, poisson, ratio =  0.5, 1, 0.1

    scale_factor = 1

if dataset_name in ['SimuMix3D_382', 'ZeroShotDeconvNet']: # 20 (train)/0 (test)
    ker_size_fp, ker_size_bp = 101, 101
    kernel_size_fp = [ker_size_fp, 101, 101]
    kernel_size_bp = [ker_size_bp, 101, 101]

    id_range   = [0, 1]
    # id_range   = [0, 1000]
    training_data_size = id_range[1] - id_range[0]
    batch_size = training_data_size
    # epochs = 250
    # epochs = 5
    # epochs = 5000

    # lamb = 642
    lamb = 560
    std_gauss, poisson, scale_factor, ratio = 0.5, 1, 1, 1

if dataset_name in ['Microtubule', 'Microtubule2', 'Nuclear_Pore_complex',\
    'Nuclear_Pore_complex2']: 
    ker_size_fp, ker_size_bp = 3, 3
    kernel_size_fp = [ker_size_fp, 31, 31]
    kernel_size_bp = [ker_size_bp, 31, 31]

    # training_data_size = 225 # 128x128
    # batch_size, epochs = 16, 360 # 225 (train)

    id_range = [0, 4]
    training_data_size = id_range[1] - id_range[0]
    batch_size = training_data_size
    epochs = 5000 # 1 (train)

    std_gauss, poisson, ratio, scale_factor = 0, 0, 1, 1

# ------------------------------------------------------------------------------
conv_mode, padding_mode, kernel_init = 'fft', 'reflect', 'gauss'
interpolation  = True
kernel_norm_fp = False
kernel_norm_bp = True
over_sampling  = 2

if dataset_dim == 2:
    std_init = [2.0, 2.0]
if dataset_dim == 3:
    std_init = [4.0, 2.0, 2.0]
# ------------------------------------------------------------------------------
# model_name = 'kernet_fp'
model_name = 'kernet'
# ------------------------------------------------------------------------------
if model_name == 'kernet_fp':
    model_suffix = '_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over{}_inter_normx_{}_ratio_{}_ts_{}_{}_s100'\
        .format(ker_size_fp, std_gauss, poisson, scale_factor, over_sampling,\
        conv_mode, ratio, id_range[0], id_range[1])
    multi_out = False
    self_supervised = False
    loss_main = torch.nn.MSELoss()

    optimizer_type = 'adam'
    # start_learning_rate = 0.0001
    start_learning_rate = 0.001
    # optimizer_type = 'lbfgs'
    # start_learning_rate = 1

    if std_gauss == 0:
        epochs = 500
    else:
        epochs = 10
    epochs = 500

if model_name == 'kernet':
    num_iter = 2
    lam = 0.0 # lambda for prior
    multi_out = False
    shared_bp = True
    self_supervised = False
    # self_supervised = True

    if self_supervised: 
        ss_marker = '_ss'
    else:
        ss_marker = ''

    model_suffix = '_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_{}_mse_over{}_inter_norm_{}_ratio_{}_ts_{}_{}{}_560'\
        .format(num_iter, ker_size_bp, std_gauss, poisson, scale_factor, lam,\
        over_sampling, conv_mode, ratio, id_range[0], id_range[1], ss_marker)

    loss_main = torch.nn.MSELoss()

    optimizer_type = 'adam'
    if self_supervised:
        start_learning_rate = 0.000001
    else:
        # start_learning_rate = 0.00001
        start_learning_rate = 0.000001
    epochs = 10000
    # start_learning_rate = 0.000001
    # epochs = 7500

# ------------------------------------------------------------------------------
warm_up = 0
use_lr_schedule = True
scheduler_cus = {}
scheduler_cus['lr']    = start_learning_rate
scheduler_cus['every'] = 2000 # 300
scheduler_cus['rate']  = 0.5
scheduler_cus['min']   = 0.00000001

# ------------------------------------------------------------------------------
if dataset_dim == 2: 
    if model_name == 'kernet':
        save_every_iter, plot_every_iter, val_every_iter = 1000, 50, 1000
        print_every_iter = 1000
    if model_name == 'kernet_fp':
        save_every_iter, plot_every_iter, val_every_iter = 5, 2, 1000
        print_every_iter = 1000

if dataset_dim == 3:
    if model_name == 'kernet':
        save_every_iter, plot_every_iter, val_every_iter = 1000, 50, 1000
        print_every_iter = 1000
    if model_name == 'kernet_fp':
        save_every_iter, plot_every_iter, val_every_iter = 5, 2, 1000
        print_every_iter = 1000

# ------------------------------------------------------------------------------
# Data 
# ------------------------------------------------------------------------------
# Training data
if dataset_name == 'tinymicro_synth': # TinyMicro (synth)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    # lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'train', 'sf_4_k_2.0_gaussian_mix_ave')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'train', 'sf_4_k_2.0_n_gaussian_std_0.03_bin_ave')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'lr.txt')
    normalization, in_channels = (False, False), 3
    training_data_size = 153024

if dataset_name == 'tinymicro_real': # TinyMicro (real)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'lr.txt')
    normalization, in_channels, scale_factor = (False, False), 3, 4
    training_data_size = 153024

if dataset_name == 'lung3_synth': # Lung3 (synth)
    hr_root_path = os.path.join(root_path, 'Lung3', 'data_transform')
    # lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'train', 'sf_1_k_2.0_n_none_std_0_bin_ave')
    # lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'train', 'sf_1_k_2.0_n_gaussian_std_0.03_bin_ave')
    lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'train', 'sf_4_k_2.0_n_none_std_0_bin_ave')
    # lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'train', 'sf_4_k_2.0_n_gaussian_std_0.03_bin_ave')

    hr_txt_file_path = os.path.join(root_path, 'Lung3', 'train.txt') 
    lr_txt_file_path = os.path.join(root_path, 'Lung3', 'train.txt') 
    normalization, in_channels, scale_factor = (False, False), 1, 4
    training_data_size = 6503

if dataset_name in ['F-actin_Nonlinear', 'Microtubules2']:
    path = os.path.join(root_path, 'BioSR', dataset_name)

    hr_root_path = os.path.join(path, f'gt_sf_{scale_factor}')
    lr_root_path = os.path.join(path, f'raw_noise_{std_gauss}')

    hr_txt_file_path = os.path.join(path, 'train.txt') 
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1

if dataset_name in ['SimuBeads3D_128', 'SimuMix3D_128', 'SimuMix3D_256']:
    path = os.path.join(root_path, 'RLN', dataset_name)
    hr_root_path = os.path.join(path, 'gt')
    lr_root_path = os.path.join(path,\
        'raw_psf_{}_gauss_{}_poiss_{}_sf_{}_ratio_{}'\
        .format(ker_size_fp, std_gauss, poisson, scale_factor, ratio))

    hr_txt_file_path = os.path.join(path, 'train.txt') 
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1

if dataset_name in ['SimuMix3D_382']:
    path = os.path.join(root_path, 'RLN', dataset_name)

    hr_root_path = os.path.join(path, 'gt')
    lr_root_path = os.path.join(path,\
        'raw_psf_101_gauss_{}_poiss_{}_sf_{}_ratio_{}_lambda_{}'\
            .format(std_gauss, poisson, scale_factor, ratio, lamb))

    hr_txt_file_path = os.path.join(path, 'train.txt') 
    lr_txt_file_path = hr_txt_file_path
    normalization, in_channels = (False, False), 1

if dataset_name in ['Microtubule', 'Microtubule2', 'Nuclear_Pore_complex',\
    'Nuclear_Pore_complex2']:
    path = os.path.join(root_path, 'RCAN3D', 'Confocal_2_STED', dataset_name)

    # hr_root_path = os.path.join(path, 'gt_128x128_0')
    # lr_root_path = os.path.join(path, 'raw_128x128_0')
    # hr_txt_file_path = os.path.join(path, 'train_128x128_0.txt')

    # hr_root_path = os.path.join(path, 'gt_1024x1024')
    # lr_root_path = os.path.join(path, 'raw_1024x1024')
    # hr_txt_file_path = os.path.join(path, 'train_1024x1024.txt')

    hr_root_path = os.path.join(path, 'gt_512x512')
    lr_root_path = os.path.join(path, 'raw_512x512')
    hr_txt_file_path = os.path.join(path, 'train_512x512.txt')

    lr_txt_file_path = hr_txt_file_path
    normalization, in_channels = (False, False), 1

if dataset_name in ['ZeroShotDeconvNet']:
    if lamb == 642: path = os.path.join(root_path, dataset_name,\
        '3D time-lapsing data_LLSM_Mitosis_H2B', str(lamb))
    if lamb == 560: path = os.path.join(root_path, dataset_name,\
        '3D time-lapsing data_LLSM_Mitosis_Mito', str(lamb))

    hr_root_path = os.path.join(path, 'raw')
    lr_root_path = hr_root_path

    hr_txt_file_path = os.path.join(path, 'raw.txt') 
    lr_txt_file_path = hr_txt_file_path
    normalization, in_channels = (False, False), 1

print('>> Load datasets from:', lr_root_path)

# ------------------------------------------------------------------------------
# Training data
training_data = dataset_utils.SRDataset(hr_root_path=hr_root_path,\
    lr_root_path=lr_root_path, hr_txt_file_path=hr_txt_file_path,\
    lr_txt_file_path=lr_txt_file_path, normalization=normalization,\
    id_range=id_range)

train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size,\
    shuffle=True, num_workers=num_workers)

# ------------------------------------------------------------------------------
# Validation data
if validation_enable == True:
    validation_data = dataset_utils.SRDataset(hr_root_path=hr_root_path,\
        lr_root_path=lr_root_path, hr_txt_file_path=hr_txt_file_path,\
        lr_txt_file_path=lr_txt_file_path, normalization=normalization,\
        id_range=[id_range[1], -1])

    valid_dataloader = DataLoader(dataset=validation_data,\
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
if model_name == 'kernet':
    FP, BP = None, None
    # FP_type, BP_type = 'pre-trained', None
    FP_type, BP_type = 'known', None

    if dataset_name in ['Microtubule', 'Microtubule2', 'Nuclear_Pore_complex',\
        'Nuclear_Pore_complex2', 'F-actin_Nonlinear', 'Microtubules2']:
        FP_type, BP_type = 'pre-trained', None
        print('pre-trained forward kernel, and to learn backward kernel.')

    # --------------------------------------------------------------------------
    if FP_type == 'pre-trained':
        print('>> Pred-trained PSF')

        # load FP parameters
        FP = kernelnet.ForwardProject(dim=dataset_dim,\
            in_channels=in_channels, scale_factor=scale_factor,\
            kernel_size=kernel_size_fp, std_init=std_init,\
            padding_mode=padding_mode, init=kernel_init, trainable=False,\
            interpolation=interpolation, kernel_norm=kernel_norm_fp,\
            over_sampling=over_sampling, conv_mode=conv_mode)

        if dataset_name == 'SimuMix3D_128':
            FP_path = os.path.join('checkpoints', dataset_name,\
                'kernet_fp_bs_{}_lr_0.01_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
                .format(batch_size, ker_size_fp, std_gauss, scale_factor),\
                'epoch_10000.pt') # 10000 (NF), 5000 (N)

        if dataset_name == 'SimuMix3D_256':
            FP_path = os.path.join('checkpoints', dataset_name,\
                'kernet_fp_bs_{}_lr_0.0005_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over4_inter_norm_fft_ratio_{}_ts_0_1'\
                .format(batch_size, ker_size_fp, std_gauss, poisson,\
                scale_factor, ratio),\
                'epoch_1000.pt')

        if dataset_name == 'Microtubule':
            FP_path = os.path.join('checkpoints', dataset_name,\
                'kernet_fp_bs_16_lr_0.0001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
                .format(ker_size_fp, std_gauss, scale_factor),\
                'epoch_5000.pt')
        
        if dataset_name == 'Nuclear_Pore_complex':
            FP_path = os.path.join('checkpoints', dataset_name,\
                'kernet_fp_bs_16_lr_0.0001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
                .format(ker_size_fp, std_gauss, scale_factor),\
                'epoch_5000.pt')

        if dataset_name == 'Nuclear_Pore_complex2':
            FP_path = os.path.join('checkpoints', dataset_name, 'forward',\
                'kernet_fp_bs_4_lr_0.01_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100'\
                .format(ker_size_fp, std_gauss, poisson, scale_factor),\
                'epoch_500.pt')

        if dataset_name == 'Microtubule2':
            FP_path = os.path.join('checkpoints', dataset_name, 'forward',\
                'kernet_fp_bs_4_lr_0.01_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100'\
                .format(ker_size_fp, std_gauss, poisson, scale_factor),\
                'epoch_500.pt')
        
        if dataset_name == 'F-actin_Nonlinear':
            FP_path = os.path.join('checkpoints', dataset_name, 'forward',\
                # 'kernet_fp_bs_1_lr_0.001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm_fft_ratio_{}'\
                'kernet_fp_bs_1_lr_0.001_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_{}_ts_0_1_s100'\
                .format(ker_size_fp, std_gauss, poisson, scale_factor, ratio),\
                # 'epoch_5000.pt')
                'epoch_500.pt')

        if dataset_name == 'Microtubules2':
            FP_path = os.path.join('checkpoints', dataset_name, 'forward',\
                # 'kernet_fp_bs_1_lr_0.001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm_fft_ratio_{}_0'\
                'kernet_fp_bs_1_lr_0.001_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_{}_ts_0_1_s100'\
                .format(ker_size_fp, std_gauss, poisson, scale_factor, ratio),\
                # 'epoch_5000.pt')
                'epoch_500.pt')

        FP_para = torch.load(FP_path, map_location=device)
        FP.load_state_dict(FP_para['model_state_dict'])
        FP.eval()

        print('>> Load from: ', FP_path)

    if FP_type == 'known':
        print('>> Known PSF')
        if dataset_dim == 2:
            ks, std = 25, 2.0
            ker = kernelnet.gauss_kernel_2d(shape=[ks, ks], std=std)\
                .to(device=device)
            ker = ker.repeat(repeats=(in_channels, 1, 1, 1))
            padd_fp = lambda x: torch.nn.functional.pad(input=x,\
                pad=(ks//2, ks//2, ks//2, ks//2), mode=padding_mode)
            conv_fp = lambda x: torch.nn.functional.conv2d(input=padd_fp(x),\
                weight=ker, groups=in_channels)
            FP = lambda x: torch.nn.functional.avg_pool2d(conv_fp(x),\
                kernel_size=25, stride=scale_factor)

        if dataset_dim == 3:
            if dataset_name == 'ZeroShotDeconvNet': 
                psf_path = os.path.join(lr_root_path, 'PSF_odd.tif')
            else:
                psf_path = os.path.join(lr_root_path, 'PSF.tif')
            PSF_true = io.imread(psf_path).astype(np.float32)
            PSF_true = torch.tensor(PSF_true[None, None]).to(device=device)
            PSF_true = torch.round(PSF_true, decimals=16)
            ks = PSF_true.shape
            padd_fp = lambda x: torch.nn.functional.pad(input=x,\
                pad=(ks[-1]//2, ks[-1]//2,\
                     ks[-2]//2, ks[-2]//2,\
                     ks[-3]//2, ks[-3]//2), mode=padding_mode)
            if conv_mode == 'direct':
                conv_fp = lambda x: torch.nn.functional.conv3d(\
                    input=padd_fp(x), weight=PSF_true, groups=in_channels)
            if conv_mode == 'fft':
                conv_fp = lambda x: fft_conv(signal=padd_fp(x),\
                    kernel=PSF_true, groups=in_channels)
            FP = lambda x: torch.nn.functional.avg_pool3d(conv_fp(x),\
                kernel_size=scale_factor, stride=scale_factor)

            print('>> Load from :', psf_path)
    
    # --------------------------------------------------------------------------
    model = kernelnet.KernelNet(dim=dataset_dim, in_channels=in_channels,\
        scale_factor=scale_factor, num_iter=num_iter,\
        kernel_size_fp=kernel_size_fp, kernel_size_bp=kernel_size_bp,\
        std_init=std_init, init=kernel_init, FP=FP, BP=BP, lam=lam,\
        padding_mode=padding_mode, multi_out=multi_out,\
        interpolation=interpolation, kernel_norm=kernel_norm_bp,\
        over_sampling=over_sampling, shared_bp=shared_bp,\
        self_supervised=self_supervised, conv_mode=conv_mode).to(device)
    
# ------------------------------------------------------------------------------
if model_name == 'kernet_fp':
    model = kernelnet.ForwardProject(dim=dataset_dim, in_channels=in_channels,\
        scale_factor=scale_factor, kernel_size=kernel_size_fp,\
        std_init=std_init, init=kernel_init, padding_mode=padding_mode,\
        trainable=True, kernel_norm=kernel_norm_fp,interpolation=interpolation,\
        conv_mode=conv_mode, over_sampling=over_sampling).to(device)

# ------------------------------------------------------------------------------
eva.count_parameters(model)
print(model)

# ------------------------------------------------------------------------------
# save
if model_name == 'kernet_fp':
    path_model = os.path.join(path_checkpoint, dataset_name, 'forward',\
        '{}_bs_{}_lr_{}{}'\
        .format(model_name, batch_size, start_learning_rate, model_suffix))

if model_name == 'kernet':
    path_model = os.path.join(path_checkpoint, dataset_name, 'backward',\
        '{}_bs_{}_lr_{}{}'\
        .format(model_name, batch_size, start_learning_rate, model_suffix))

writer = SummaryWriter(os.path.join(path_model, 'log'))
print('>> Save model to', path_model)

# ------------------------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------------------------
if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=start_learning_rate)
if optimizer_type == 'lbfgs':
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=start_learning_rate)
    optimizer = torch.optim.LBFGS(model.parameters(),lr=start_learning_rate,\
        line_search_fn='strong_wolfe')

print('>> Start training ... ')
print(time.asctime(time.localtime(time.time())))

num_batches     = len(train_dataloader)
num_batches_val = 0
if validation_enable == True: 
    num_batches_val = len(valid_dataloader)

print('>> Number of training batches: {}, validation batches: {}'\
    .format(num_batches, num_batches_val))

if self_supervised == True:
    print('Training under self-supervised mode.')

if training_data_size == 1:
    sample = training_data[0]
    x, y = sample['lr'].to(device)[None], sample['hr'].to(device)[None]
    y = y * ratio
else:
    x, y = [], []
    for i in range(training_data_size):
        sample = training_data[i]
        x.append(sample['lr'])
        y.append(sample['hr'])
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    y = y * ratio

for i_epoch in range(epochs):
    print('\n'+'-'*98)
    print('Epoch: {}/{} | Batch size: {} | Num of Batches: {}'\
        .format(i_epoch + 1, epochs, batch_size, num_batches))
    print('-'*98)
    # --------------------------------------------------------------------------
    ave_ssim, ave_psnr = 0, 0
    print_loss, print_ssim, print_psnr = [], [], []
    
    start_time = time.time()
    # --------------------------------------------------------------------------
    model.train()
    # for i_batch, sample in enumerate(train_dataloader):
    for i_batch in range(num_batches):
        i_iter = i_batch + i_epoch * num_batches # index of iteration
        # ----------------------------------------------------------------------
        # load data
        # x, y = sample['lr'].to(device), sample['hr'].to(device)
        # y = y * ratio

        if model_name == 'kernet_fp': inpt, gt = y, x
        if model_name == 'kernet':
            if self_supervised == True: 
                inpt, gt = x, x
            else:
                inpt, gt = x, y

        # ----------------------------------------------------------------------
        # optimize
        if optimizer_type == 'lbfgs':
            # L-BFGS
            loss = 0.0
            pred = 0.0
            def closure():
                global loss
                global pred
                pred = model(inpt)
                optimizer.zero_grad()
                loss = loss_main(pred, gt)
                loss.backward()
                return loss
            optimizer.step(closure) 

        else:
            optimizer.zero_grad()
            pred = model(inpt)
            loss = loss_main(pred, gt)
            loss.backward()
            optimizer.step()

        # ----------------------------------------------------------------------
        # custom learning rate scheduler
        if use_lr_schedule == True:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus['lr']
                # set learning rate
                for g in optimizer.param_groups: g['lr'] = lr 

            if (i_iter >= warm_up):
                if (i_iter + 1 - warm_up) % scheduler_cus['every'] == 0:
                    lr = scheduler_cus['lr'] * (scheduler_cus['rate']**\
                        ((i_iter + 1 - warm_up) // scheduler_cus['every']))
                    lr = np.maximum(lr, scheduler_cus['min'])
                    for g in optimizer.param_groups: g['lr'] = lr 
        else:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus['lr']
                for g in optimizer.param_groups: g['lr'] = lr 

            if (i_iter >= warm_up):
                for g in optimizer.param_groups: g['lr'] = scheduler_cus['lr']

        # ----------------------------------------------------------------------
        if multi_out == False: out = pred
        if multi_out == True:  out = pred[-1]

        # ----------------------------------------------------------------------
        # plot loss and metrics
        if i_iter % plot_every_iter == 0:
            if dataset_dim == 2: ave_ssim, ave_psnr = eva.measure_2d(\
                img_test=out, img_true=gt, data_range=data_range)
            if dataset_dim == 3: ave_ssim, ave_psnr = eva.measure_3d(\
                img_test=out, img_true=gt, data_range=data_range)
            if writer != None:
                writer.add_scalar('loss', loss, i_iter)
                writer.add_scalar('psnr', ave_psnr, i_iter)
                writer.add_scalar('ssim', ave_ssim, i_iter)
                writer.add_scalar('Leanring Rate',\
                    optimizer.param_groups[-1]['lr'], i_iter)
            # if (i_iter > 5000) & (ave_psnr < 10.0):
            #     print('\nPSNR ({:>.4f}) is too low, break!'.format(ave_psnr))
            #     writer.flush()
            #     writer.close()
            #     os._exit(0)

        # ----------------------------------------------------------------------
        # print and save model
        if dataset_dim == 2: s, p = eva.measure_2d(img_test=out, img_true=gt,\
            data_range=data_range)
        if dataset_dim == 3: s, p = eva.measure_3d(img_test=out, img_true=gt,\
            data_range=data_range)
        print_loss.append(loss.cpu().detach().numpy())
        print_ssim.append(s)
        print_psnr.append(p)
        print('#', end='')

        if i_iter % print_every_iter == 0:
            print('\nEpoch: {}, Iter: {}, loss: {:>.5f}, PSNR: {:>.5f},\
                SSIM: {:>.5f}'.\
                format(i_epoch, i_iter, np.mean(print_loss),\
                    np.mean(print_psnr), np.mean(print_ssim)))
            print('Computation time: {:>.2f} s'.format(time.time()-start_time))
            start_time = time.time()
            print_loss, print_ssim, print_psnr = [], [], []

        # ----------------------------------------------------------------------
        # save model and relative information
        if i_iter % save_every_iter == 0:
            print('\nSave model ... (Epoch: {}, Iteration: {})'\
                .format(i_epoch, i_iter))
            model_dict = {'model_state_dict': model.state_dict()}
            torch.save(model_dict, os.path.join(path_model, 'epoch_{}.pt'\
                .format(i_iter)))

        # ----------------------------------------------------------------------
        # validation
        if (i_iter % val_every_iter == 0) and (validation_enable == True):
            print('validation ...')
            running_val_loss, running_val_ssim, running_val_psnr = 0, 0, 0
            model.eval()
            for i_batch_val, sample_val in enumerate(valid_dataloader):
                x_val = sample_val['lr'].to(device)
                y_val = sample_val['hr'].to(device)
                if model_name == 'kernel_fp': inpt, gt = y_val, x_val
                if model_name == 'kernet':    inpt, gt = x_val, y_val

                pred_val = model(inpt)
                loss_val = loss_main(pred_val, gt)

                if multi_out == True:  out_val = pred_val[-1]
                if multi_out == False: out_val = pred_val

                if dataset_dim == 2: ave_ssim, ave_psnr = eva.measure_2d(\
                    img_test=out_val, img_true=gt, data_range=data_range)
                if dataset_dim == 3: ave_ssim, ave_psnr = eva.measure_3d(\
                    img_test=out_val, img_true=gt, data_range=data_range)

                running_val_loss += loss_val.cpu().detach().numpy()
                running_val_psnr += ave_psnr
                running_val_ssim += ave_ssim
                print('#', end='')

            print('\nValidation, Loss: {:>.5f}, PSNR: {:>.5f}, SSIM: {:>.5f}'\
                .format(running_val_loss / num_batches_val,\
                        running_val_psnr / num_batches_val,\
                        running_val_ssim / num_batches_val))

            if writer != None:
                writer.add_scalar('loss_val',\
                    running_val_loss / num_batches_val, i_iter)
                writer.add_scalar('psnr_val',\
                    running_val_psnr / num_batches_val, i_iter)
                writer.add_scalar('ssim_val',\
                    running_val_ssim / num_batches_val, i_iter)
            model.train()

# ------------------------------------------------------------------------------
# save the last one model
print('\nSave model ... (Epoch: {}, Iteration: {})'.format(i_epoch, i_iter+1))
model_dict = {'model_state_dict': model.state_dict()}
torch.save(model_dict, os.path.join(path_model, 'epoch_{}.pt'\
    .format(i_iter + 1)))

# ------------------------------------------------------------------------------
writer.flush() 
writer.close()
print('Training done!')