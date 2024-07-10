import numpy as np
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
# dataset_name = 'SimuBeads3D_128'
# dataset_name = 'SimuMix3D_128'
# dataset_name = 'SimuMix3D_128_2'
# dataset_name = 'SimuMix3D_128_3'
dataset_name = 'SimuMix3D_256'
# dataset_name = 'SimuMix3D_256_2'
# dataset_name = 'SimuMix3D_256s'
# dataset_name = 'SimuMix3D_382'

# ------------------------------------------------------------------------------
if dataset_name in ['SimuBeads3D_128', 'SimuMix3D_128', 'SimuMix3D_128_2',\
    'SimuMix3D_128_3','SimuMix3D_256','SimuMix3D_256_2', 'SimuMix3D_256s']:\
    lamb = None
if dataset_name in ['SimuMix3D_382']:
    # lamb = 642 # wavelength of the emission light
    lamb = 560

path_dataset    = os.path.join('F:', os.sep, 'Datasets', 'RLN', dataset_name)
path_dataset_gt = os.path.join(path_dataset, 'gt')

# ------------------------------------------------------------------------------
if lamb == None:
    path_psf = os.path.join(path_dataset, 'PSF.tif')
else:
    path_psf = os.path.join(path_dataset, f'PSF_{lamb}.tif')

print('>> Load PSF from:', path_psf)

# PSF
PSF     = io.imread(path_psf).astype(np.float32)
PSF_odd = utils_data.even2odd(PSF)
print('>> PSF shape from {} to {}'.format(PSF.shape, PSF_odd.shape))

# crop PSF
if dataset_name in ['SimuBeads3D_128', 'SimuMix3D_128']:
    s_crop = 127
    # s_crop = 63
    PSF_crop = utils_data.center_crop(PSF_odd, size=(s_crop,127,127))

if dataset_name in ['SimuMix3D_128_3','SimuMix3D_256_2','SimuMix3D_256s']:
    s_crop = 63
    PSF_crop = utils_data.center_crop(PSF_odd, size=(s_crop, 31, 31))

if dataset_name in ['SimuMix3D_256']:
    s_crop = 31
    PSF_crop = utils_data.center_crop(PSF_odd, size=(s_crop,)*3)

if dataset_name in ['SimuMix3D_382']:
    s_crop = 101
    PSF_crop = utils_data.center_crop(PSF_odd, size=(101, s_crop, s_crop))

print('>> PSF after crop:', PSF_crop.shape, ', sum =', PSF_crop.sum())
PSF_crop = PSF_crop/PSF_crop.sum()

# ------------------------------------------------------------------------------
# single image
data_gt_single = io.imread(os.path.join(path_dataset_gt, '1.tif'))
data_gt_single = data_gt_single.astype(np.float32)
print('GT shape:', data_gt_single.shape)

# ------------------------------------------------------------------------------
# multiple images
# ------------------------------------------------------------------------------
# id_data = range(1, 101, 1)
id_data = range(1, 21, 1)
# id_data = [1]

std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0, 0, 1
scale_factor = 1

# ------------------------------------------------------------------------------
# save to
if lamb == None:
    path_dataset_blur = os.path.join(path_dataset,\
        'raw_psf_{}_gauss_{}_poiss_{}_sf_{}_ratio_{}'\
        .format(s_crop, std_gauss, poisson, scale_factor, ratio))
else:
    path_dataset_blur = os.path.join(path_dataset,\
        'raw_psf_{}_gauss_{}_poiss_{}_sf_{}_ratio_{}_lambda_{}'\
        .format(s_crop, std_gauss, poisson, scale_factor, ratio, lamb))

if not os.path.exists(path_dataset_blur): 
    os.makedirs(path_dataset_blur)
print('>> save to:', path_dataset_blur)

io.imsave(os.path.join(path_dataset_blur, 'PSF.tif'), arr=PSF_crop,\
    check_contrast=False)
# ------------------------------------------------------------------------------

for i in id_data:
    data_gt   = io.imread(os.path.join(path_dataset_gt, f'{i}.tif'))

    # scale to control SNR
    data_gt   = data_gt.astype(np.float32) * ratio
    data_blur = dcv.Convolution(data_gt, PSF_crop, padding_mode='reflect',\
                                domain='fft')

    # add noise
    data_blur_n = utils_data.add_mix_noise(data_blur, poisson=poisson,\
            sigma_gauss=std_gauss, scale_factor=scale_factor)

    # SNR
    print(f'sample [{i}],', 'SNR:', eva.SNR(data_blur, data_blur_n))
    io.imsave(os.path.join(path_dataset_blur, f'{i}.tif'),\
        arr=data_blur_n, check_contrast=False)