import torch, os, time
import numpy as np
import skimage.io as io
import methods.deconvolution as dcv
from utils import dataset_utils as utils_data
from models import kernelnet
from fft_conv_pytorch import fft_conv

# ------------------------------------------------------------------------------
# Parameter setting
# ------------------------------------------------------------------------------
root_path = os.path.join('F:', os.sep, 'Datasets')
device = torch.device('cpu')

# ------------------------------------------------------------------------------
# simulation data set
# dataset_name_train, dataset_name_test = 'SimuMix3D_128', 'SimuBeads3D_128'
# dataset_name_train, dataset_name_test = 'SimuMix3D_128', 'SimuMix3D_128'
# dataset_name_train, dataset_name_test = 'SimuMix3D_256', 'SimuMix3D_256'
# dataset_name_train, dataset_name_test = 'SimuMix3D_382', 'SimuMix3D_382'
# ------------------------------------------------------------------------------
# confocal/STED volume data set
# dataset_name_train, dataset_name_test = 'Microtubule', 'Microtubule'
# dataset_name_train, dataset_name_test = 'Microtubule2', 'Microtubule2'
# dataset_name_train, dataset_name_test = 'Nuclear_Pore_complex', 'Nuclear_Pore_complex'
# dataset_name_train, dataset_name_test = 'Nuclear_Pore_complex2', 'Nuclear_Pore_complex2'
# ------------------------------------------------------------------------------
# BioSR data set
# dataset_name_train, dataset_name_test = 'F-actin_Nonlinear', 'F-actin_Nonlinear'
# dataset_name_train, dataset_name_test = 'Microtubules2', 'Microtubules2'
# ------------------------------------------------------------------------------
# LLSM volume data set
dataset_name_train, dataset_name_test = 'SimuMix3D_382', 'ZeroShotDeconvNet'
# dataset_name_train, dataset_name_test = 'ZeroShotDeconvNet', 'ZeroShotDeconvNet'

# ------------------------------------------------------------------------------
if dataset_name_train in ['SimuBeads3D_128', 'SimuMix3D_128', 'SimuMix3D_256']:
    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = [ker_size_fp, 31, 31]
    kernel_size_bp = [ker_size_bp, 25, 25]
    dim = 3

if dataset_name_train in ['SimuMix3D_382', 'ZeroShotDeconvNet']:
    ker_size_fp, ker_size_bp = 101, 101
    kernel_size_fp = [ker_size_fp, 101, 101]
    kernel_size_bp = [ker_size_bp, 101, 101]
    dim = 3

    # wave_length = '642'
    wave_length = '560'

if dataset_name_train in ['Microtubule', 'Microtubule2', 'Nuclear_Pore_complex', 'Nuclear_Pore_complex2']:
    ker_size_fp, ker_size_bp = 3, 3
    kernel_size_fp = [ker_size_fp, 31, 31]
    kernel_size_bp = [ker_size_bp, 31, 31]
    dim = 3

if dataset_name_train in ['F-actin_Nonlinear', 'Microtubules2']:
    # ker_size_fp, ker_size_bp = 101, 101
    ker_size_fp, ker_size_bp = 31, 31
    kernel_size_fp = (ker_size_fp,)*2
    kernel_size_bp = (ker_size_bp,)*2
    dim = 2

# ------------------------------------------------------------------------------
# num_sample = 20
num_sample = 1000
# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = [0, 1, 2, 3, 4, 5]
# id_sample = range(0, 1000, 4)
# id_sample = [0, 1, 2, 3, 4, 5, 6] 
# id_sample = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# id_sample = [0, 1, 2, 3, 4, 5, 6]
id_sample = [0]

# suffix_net = '_ss'
suffix_net = ''

# ------------------------------------------------------------------------------
std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0, 0, 1
# std_gauss, poisson, ratio = 9, 1, 1
# ------------------------------------------------------------------------------
FP_type, BP_type = 'known', 'learned'
# FP_type, BP_type = 'known', 'known'
# FP_type, BP_type = 'pre-trained', 'learned'
# FP_type, BP_type = 'pre-trained', 'known'
# ------------------------------------------------------------------------------
num_iter_train = 2
num_iter_test  = num_iter_train + 0
# ------------------------------------------------------------------------------
eps = 0.000001
scale_factor   = 1
interpolation  = True
kernel_norm_fp = False
kernel_norm_bp = True
over_sampling  = 2
padding_mode   = 'reflect'
if dim == 3:
    std_init = [4.0, 2.0, 2.0]
if dim == 2:
    std_init = [2.0, 2.0]
shared_bp = True
conv_mode = 'fft'
# ------------------------------------------------------------------------------
if dataset_name_test in ['ZeroShotDeconvNet']:
    path_fig = os.path.join('outputs', 'figures', dataset_name_test.lower(),\
        'Mitosis', wave_length)
else:
    path_fig = os.path.join('outputs', 'figures', dataset_name_test.lower(),\
        f'scale_{scale_factor}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}')

if not os.path.exists(path_fig): os.makedirs(path_fig, exist_ok=True)

# ------------------------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------------------------
if dataset_name_test in ['SimuMix3D_128', 'SimuBeads3D_128', 'SimuMix3D_256']:
    path = os.path.join(root_path, 'RLN', dataset_name_test)
    hr_root_path = os.path.join(path, 'gt')
    lr_root_path = os.path.join(path,\
        'raw_psf_31_gauss_{}_poiss_{}_sf_{}_ratio_{}'\
        .format(std_gauss, poisson, scale_factor, ratio))

    hr_txt_file_path = os.path.join(path, 'test.txt')
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1
    PSF_true = io.imread(os.path.join(lr_root_path, 'PSF.tif'))

if dataset_name_test in ['SimuMix3D_382']:
    path = os.path.join(root_path, 'RLN', dataset_name_test)
    hr_root_path = os.path.join(path, 'gt')
    lr_root_path = os.path.join(path,\
        f'raw_psf_101_noise_{std_gauss}_sf_{scale_factor}_ratio_{ratio}')

    hr_txt_file_path = os.path.join(path, 'test.txt')
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1
    PSF_true = io.imread(os.path.join(lr_root_path, 'PSF.tif'))

if dataset_name_test in ['ZeroShotDeconvNet']:
    if wave_length == '642':
        path = os.path.join(root_path, dataset_name_test,\
            '3D time-lapsing data_LLSM_Mitosis_H2B', '642')
    if wave_length == '560':
        path = os.path.join(root_path, dataset_name_test,\
            '3D time-lapsing data_LLSM_Mitosis_Mito', '560')
    hr_root_path = os.path.join(path, 'raw')
    lr_root_path = os.path.join(path, 'raw')

    hr_txt_file_path = os.path.join(path, 'raw.txt')
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1
    PSF_true = io.imread(os.path.join(lr_root_path, 'PSF_odd.tif'))

if dataset_name_test in ['Microtubule', 'Microtubule2', 'Nuclear_Pore_complex', 'Nuclear_Pore_complex2']:
    path = os.path.join(root_path, 'RCAN3D', 'Confocal_2_STED',\
        dataset_name_test)
    hr_root_path = os.path.join(path, 'gt_1024x1024')
    lr_root_path = os.path.join(path, 'raw_1024x1024')

    hr_txt_file_path = os.path.join(path, 'test_1024x1024.txt')
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1
    PSF_true = np.zeros(shape=kernel_size_fp)

    conv_mode = 'direct'
    std_gauss, poisson, ratio = 0, 0, 1

if dataset_name_test in ['F-actin_Nonlinear', 'Microtubules2']:
    path = os.path.join(root_path, 'BioSR', dataset_name_test)
    hr_root_path = os.path.join(path, f'gt_sf_{scale_factor}')
    lr_root_path = os.path.join(path, f'raw_noise_{std_gauss}')

    hr_txt_file_path = os.path.join(path, 'test.txt')
    lr_txt_file_path = hr_txt_file_path

    normalization, in_channels = (False, False), 1
    PSF_true = np.zeros(shape=kernel_size_fp)

PSF_true = PSF_true.astype(np.float32)

# ------------------------------------------------------------------------------
print('-'*80)
print('load data from:', lr_root_path)

dataset_test = utils_data.SRDataset(hr_root_path=hr_root_path,\
    lr_root_path=lr_root_path, hr_txt_file_path=hr_txt_file_path,\
    lr_txt_file_path=lr_txt_file_path, normalization=normalization,\
    id_range=[0, num_sample])

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
FP, BP = None, None
# Forward Projection
print('-'*80)
if FP_type == 'pre-trained':
    print('FP kernel (PSF) (Pre-trained)')
    if dataset_name_train == 'SimuMix3D_128':
        num_data  = 2
        id_repeat = 1
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            # 'kernet_fp_bs_1_lr_0.0001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
            'kernet_fp_bs_2_lr_1_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_{}_ts_0_2_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor, ratio),\
            # 'epoch_5000.pt') # 10000 (NF), 5000 (N)
            'epoch_20.pt') # 10000 (NF), 5000 (N)
    
    if dataset_name_train == 'SimuMix3D_256':
        num_data  = 1
        id_repeat = 3
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            'kernet_fp_bs_1_lr_1_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_{}_ts_3_4_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor, ratio),\
            'epoch_10.pt') # 50 (NF), 10 (N)

    if dataset_name_train == 'Microtubule':
        FP_path = os.path.join('checkpoints', dataset_name_train,\
            'kernet_fp_bs_16_lr_0.0001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
            .format(ker_size_fp, std_gauss, scale_factor),\
            'epoch_5000.pt')
    
    if dataset_name_train == 'Nuclear_Pore_complex':
        FP_path = os.path.join('checkpoints', dataset_name_train,\
            'kernet_fp_bs_16_lr_0.0001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm'\
            .format(ker_size_fp, std_gauss, scale_factor),\
            'epoch_5000.pt')

    if dataset_name_train == 'Nuclear_Pore_complex2':
        num_data  = 1
        id_repeat = 1
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            'kernet_fp_bs_4_lr_0.01_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor),\
            'epoch_500.pt')

    if dataset_name_train == 'Microtubule2':
        num_data  = 1
        id_repeat = 1
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            'kernet_fp_bs_4_lr_0.01_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_4_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor),\
            'epoch_500.pt')

    if dataset_name_train == 'F-actin_Nonlinear':
        num_data  = 1
        id_repeat = 1
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            # 'kernet_fp_bs_1_lr_0.001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm_fft_ratio_1'\
            'kernet_fp_bs_1_lr_0.001_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor),\
            # 'epoch_5000.pt')
            'epoch_500.pt')

    if dataset_name_train == 'Microtubules2':
        num_data  = 1
        id_repeat = 1
        FP_path = os.path.join('checkpoints', dataset_name_train, 'forward',\
            # 'kernet_fp_bs_1_lr_0.001_ker_{}_noise_{}_sf_{}_mse_over2_inter_norm_fft_ratio_1_0'\
            'kernet_fp_bs_1_lr_0.001_ker_{}_gauss_{}_poiss_{}_sf_{}_mse_over2_inter_normx_fft_ratio_1_ts_0_1_s100'\
            .format(ker_size_fp, std_gauss, poisson, scale_factor),\
            # 'epoch_5000.pt')
            'epoch_500.pt')

    # --------------------------------------------------------------------------
    print('model: ', FP_path)
    FP = kernelnet.ForwardProject(dim=dim, in_channels=in_channels,\
        scale_factor=scale_factor, kernel_size=kernel_size_fp,\
        std_init=std_init, init='gauss', kernel_norm=kernel_norm_fp,\
        padding_mode=padding_mode, interpolation=interpolation,\
        over_sampling=over_sampling, conv_mode=conv_mode).to(device)

    ker_init = FP.conv.get_kernel().detach().numpy()[0,0]
    FP.load_state_dict(torch.load(FP_path,\
        map_location=device)['model_state_dict'])
    FP.eval()

# ------------------------------------------------------------------------------
if FP_type == 'known':
    print('>> PSF (Known)')
    ks = PSF_true.shape
    weight  = torch.tensor(PSF_true[None, None]).to(device=device)
    padd_fp = lambda x: torch.nn.functional.pad(input=x,\
        pad=(ks[-1]//2, ks[-1]//2, ks[-2]//2, ks[-2]//2, ks[-3]//2, ks[-3]//2),\
        mode=padding_mode)
    if conv_mode == 'direct':
        conv_fp = lambda x: torch.nn.functional.conv3d(input=padd_fp(x),\
            weight=weight, groups=in_channels)
    if conv_mode == 'fft':
        conv_fp = lambda x: fft_conv(signal=padd_fp(x), kernel=weight,\
            groups=in_channels)
    FP = lambda x: torch.nn.functional.avg_pool3d(conv_fp(x),\
        kernel_size=scale_factor, stride=scale_factor)
    ker_FP = weight.numpy()[0, 0]
    # The PSF now is known, setting the initial PSF as all zeros.
    ker_init = np.zeros_like(ker_FP)

# ------------------------------------------------------------------------------
# Backward Projection
print('-'*80)
if BP_type == 'known':
    print('BP kernel (Known)')
    BP = lambda x: dcv.Convolution(PSF=ker_FP, x=x.detach().numpy()[0,0],\
        padding_mode=padding_mode, domain=conv_mode)
    ker_BP = PSF_true

# ------------------------------------------------------------------------------
model = kernelnet.KernelNet(in_channels=in_channels, scale_factor=scale_factor,\
    dim=dim, num_iter=num_iter_test, kernel_size_fp=kernel_size_fp,\
    kernel_size_bp=kernel_size_bp, std_init=std_init, init='gauss',\
    padding_mode=padding_mode, FP=FP, BP=BP, lam=0.0, return_inter=True,\
    multi_out=False, over_sampling=over_sampling, kernel_norm=kernel_norm_bp,\
    interpolation=interpolation, shared_bp=shared_bp, \
    conv_mode=conv_mode).to(device)

# ------------------------------------------------------------------------------
if BP_type == 'learned':
    print('BP kernel (Leanred)')
    if dataset_name_train == 'SimuMix3D_128':
        num_data  = 3
        id_repeat = 1
        model_path = os.path.join('checkpoints', dataset_name_train,\
            'backward',\
            'kernet_bs_3_lr_1e-06_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_3'\
            # 'kernet_bs_1_lr_1e-06_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_1_ss'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson, \
            scale_factor, ratio),\
            # 'epoch_5000.pt')
            'epoch_10000.pt')
    
    if dataset_name_train == 'SimuMix3D_256':
        model_path = os.path.join('checkpoints', dataset_name_train,\
            'kernet_bs_1_lr_0.0005_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over4_inter_norm_fft_ratio_{}_ts_0_1_abss_mo'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson, scale_factor, ratio),\
            'epoch_1000.pt')
    
    if dataset_name_train == 'SimuMix3D_382':
        num_data  = 9
        id_repeat = 1
        if wave_length == '642':
            model_path = os.path.join('checkpoints', dataset_name_train,\
                'backward',\
                # 'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_1_642'\
                'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_9_642'\
                .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
                scale_factor, ratio),\
                # 'epoch_6000.pt')
                'epoch_4000.pt')

        if wave_length == '560':
            model_path = os.path.join('checkpoints', dataset_name_train,\
                'backward',\
                # 'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_1_560'\
                'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_9_560'\
                .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
                scale_factor, ratio),\
                # 'epoch_6000.pt')
                'epoch_4000.pt')
    
    if dataset_name_train == 'ZeroShotDeconvNet':
        model_path = os.path.join('checkpoints', dataset_name_train,\
            # 'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_noise_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_101x101_642_ss'\
            'kernet_bs_1_lr_1e-07_iter_{}_ker_{}_noise_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_101x101_560_ss'\
            .format(num_iter_train, ker_size_bp, std_gauss, scale_factor, ratio),\
            'epoch_3000.pt')
    
    if dataset_name_train == 'Microtubule':
        model_path = os.path.join('checkpoints', dataset_name_train,\
            'kernet_bs_1_lr_1e-05_iter_{}_ker_{}_noise_{}_sf_{}_lam_0.0_mse_over{}_inter_norm_1024'\
            .format(num_iter_train, ker_size_bp, std_gauss, scale_factor, over_sampling),\
            'epoch_5000.pt')
    
    if dataset_name_train == 'Nuclear_Pore_complex':
        model_path = os.path.join('checkpoints', dataset_name_train,\
            'kernet_bs_1_lr_1e-06_iter_{}_ker_{}_noise_{}_sf_{}_lam_0.0_mse_over{}_inter_norm_1024'\
            .format(num_iter_train, ker_size_bp, std_gauss, scale_factor, over_sampling),\
            'epoch_5000.pt')
    
    if dataset_name_train == 'Nuclear_Pore_complex2':
        model_path = os.path.join('checkpoints', dataset_name_train,'backward',\
            'kernet_bs_4_lr_1e-05_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over{}_inter_norm_fft_ratio_1_ts_0_4'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
            scale_factor, over_sampling),\
            'epoch_10000.pt')

    if dataset_name_train == 'Microtubule2':
        model_path = os.path.join('checkpoints', dataset_name_train,'backward',\
            'kernet_bs_4_lr_1e-05_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over{}_inter_norm_fft_ratio_1_ts_0_4'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
            scale_factor, over_sampling),\
            'epoch_10000.pt')
    
    if dataset_name_train == 'F-actin_Nonlinear':
        model_path = os.path.join('checkpoints', dataset_name_train,'backward',\
            'kernet_bs_1_lr_1e-05_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_1'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
            scale_factor, ratio),\
            'epoch_10000.pt')

    if dataset_name_train == 'Microtubules2':
        model_path = os.path.join('checkpoints', dataset_name_train,'backward',\
            # 'kernet_bs_1_lr_1e-05_iter_{}_ker_{}_noise_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_0'\
            'kernet_bs_1_lr_0.0001_iter_{}_ker_{}_gauss_{}_poiss_{}_sf_{}_lam_0.0_mse_over2_inter_norm_fft_ratio_{}_ts_0_1'\
            .format(num_iter_train, ker_size_bp, std_gauss, poisson,\
            scale_factor, ratio),\
            # 'epoch_5000.pt')
            'epoch_10000.pt')

    print('Model: ', model_path)
    model.load_state_dict(torch.load(model_path,\
        map_location=device)['model_state_dict'], strict=False)
    model.eval()

    # get the learned BP kernel
    if shared_bp == True:
        ker_BP = model.BP.conv.get_kernel()[0, 0].detach().numpy()
    else:
        ker_BP = model.BP[0].conv.get_kernel()[0, 0].detach().numpy()

print('BP kernel shape:', ker_BP.shape)

if FP_type == 'pre-trained':
    # get the FP learned FP kernel
    ker_FP = model.FP.conv.get_kernel()[0, 0].detach().numpy()
    print('FP kernel shape:', ker_FP.shape)

# ------------------------------------------------------------------------------
# Save kernels
# ------------------------------------------------------------------------------
path_kernel = os.path.join(path_fig, f'kernels_bc_{num_data}_re_{id_repeat}')

if os.path.exists(path_kernel) is not True:
    os.makedirs(path_kernel, exist_ok=True)

imsave_data = lambda fname, arr: io.imsave(\
    fname=os.path.join(path_kernel, fname), arr=arr, check_contrast=False)

ker_FP   = utils_data.padding_kernel(ker_FP, PSF_true)
ker_BP   = utils_data.padding_kernel(ker_BP, PSF_true)
ker_init = utils_data.padding_kernel(ker_init, PSF_true)

print('-'*80)
print('save kernels ...')
print('save to:', path_kernel)
imsave_data('kernel_init.tif', ker_init)
imsave_data('kernel_fp.tif',   ker_FP)
imsave_data('kernel_true.tif', PSF_true)
imsave_data(f'kernel_bp{suffix_net}.tif',   ker_BP)

# os._exit(0)
# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
print('-'*80)
print('evaluation ...')
print('conv mode: ', conv_mode)
t2n = lambda x: x.cpu().detach().numpy()[0, 0]
id_inter_plot = [0, 1, 2, 3, 4, 5, 6] # the id of sampels to show the intermedia results

for i in id_sample:
    # load one sample
    ds = dataset_test[i]
    x = torch.unsqueeze(ds['lr'], 0).to(device)
    y = torch.unsqueeze(ds['hr'], 0).to(device) * ratio
    print('-'*80)
    print('Sample [{}], input shape:{}, GT shape:{}'\
        .format(i, list(x.shape), list(y.shape)))

    # intermedia results
    if i in id_inter_plot:
        # forward projection
        y_fp = model.FP(y) 
        x0 = torch.nn.functional.interpolate(x, scale_factor=scale_factor,\
            mode='nearest-exact')
        x0_fp = model.FP(x0)

        # backward projeciton
        if shared_bp == True:
            bp = model.BP(x/(x0_fp + eps))
        else:
            bp = model.BP[0](x/(x0_fp + eps))

        y_fp, x0, x0_fp = t2n(y_fp), t2n(x0), t2n(x0_fp)
        if BP_type == 'learned': bp = t2n(bp)

    ts = time.time()
    y_pred_all = model(x)
    print('Time : {:.2f} s, each iteration: {:.2f} s.'\
        .format(time.time()-ts, (time.time()-ts)/num_iter_test))

    y_pred_all = y_pred_all.cpu().detach().numpy()[:,0,0]
    y, x = t2n(y), t2n(x)

    # --------------------------------------------------------------------------
    # Save results
    name_net = 'kernelnet'
    path_single_data = os.path.join(path_fig, f'sample_{i}',\
        name_net + suffix_net)
    if not os.path.exists(path_single_data):\
        os.makedirs(path_single_data, exist_ok=True)
    
    imsave_sample = lambda path, fname, arr:\
        io.imsave(fname=os.path.join(path, fname), arr=arr,\
            check_contrast=False)

    print('Save result to:', path_single_data)
    if i in id_inter_plot:
        imsave_sample(path_single_data, 'x0.tif', x0)
        imsave_sample(path_single_data, 'y_fp.tif', y_fp)
        imsave_sample(path_single_data, 'x0_fp.tif', x0_fp)
        imsave_sample(path_single_data, 'bp.tif', bp)
    
    if dataset_name_test != 'ZeroShotDeconvNet':
        imsave_sample(path_single_data, 'x.tif', x)
        imsave_sample(path_single_data, 'y.tif', y)

    if dataset_name_test == 'ZeroShotDeconvNet':
        imsave_sample(path_single_data, 'y_pred_all.tif',\
            y_pred_all[-1].astype(np.uint16))
    else:
        imsave_sample(path_single_data, 'y_pred_all.tif', y_pred_all)
        # imsave_sample(path_single_data, f'y_pred_{num_iter_test}.tif',\
            # y_pred_all)
    print('-'*80)