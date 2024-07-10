import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva
# ================================================================================================================
enable_traditonal, enable_gaussian, enable_bw, enable_wb = 0, 0, 0, 1
num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb   = 30, 30, 30, 30

# ================================================================================================================
# dataset_name = 'SimuBeads3D'
# fig_path     = os.path.join('outputs', 'figures', 'simubeads3d')

# ================================================================================================================
dataset_name = 'SimuMix3D_128'
noise_level  = 0
id_data = 1
fig_path     = os.path.join('outputs', 'figures', 'simumix3d')

# ================================================================================================================
# load data
dataset_path    = os.path.join('F:', os.sep, 'Datasets', 'RLN', dataset_name)
data_gt_path    = os.path.join(dataset_path, 'ground_truth', f'{id_data}.tif')
data_input_path = os.path.join(dataset_path, f'input_noise_{noise_level}_crop', f'{id_data}.tif')
psf_path        = os.path.join(data_input_path, 'PSF.tif')

data_gt     = io.imread(data_gt_path).astype(np.float32)
data_input  = io.imread(data_input_path).astype(np.float32)
PSF         = io.imread(psf_path).astype(np.float32)

print('GT: {}, Input: {}, PSF: {}'.format(data_gt.shape, data_input.shape, PSF.shape))

# ================================================================================================================
# save result to path
for meth in ['traditional', 'gaussian', 'butterworth', 'wiener_butterworth']:
    meth_path = os.path.join(fig_path, meth)
    if not os.path.exists(meth_path): os.mkdir(meth_path)

# ================================================================================================================
# generate input
Sx, Sy, Sz = data_gt.shape

PSF = PSF / np.sum(PSF)
PSF_align = dcv.align_size(PSF, Sx, Sy, Sz)
OTF_fp = np.fft.fftn(np.fft.ifftshift(PSF_align))

# ================================================================================================================
ratio = 1.0     # control the Pisson noise level
thresh = 0.0    # used to add some background
data_gt_scale = data_gt * ratio # use this
# data_gt_scale=((1 + 1*np.random.rand()) * data_gt + thresh) * ratio
# data_gt_scale=(data_gt + thresh) * ratio
data_blur = dcv.ConvFFT3_S(data_gt_scale, OTF_fp)

# ================================================================================================================
# add noise
data_blur_poi = np.random.poisson(lam=data_blur)
max_signal = np.max(data_blur_poi)
data_blur_poi_norm = data_blur_poi / max_signal
sigma = 0.5 # control the level of Gaussian noise, constant or random value
data_blur_poi_gaus = data_blur_poi_norm + np.random.normal(loc=0, scale=sigma/max_signal, size=data_blur_poi_norm.shape)
data_blur_poi_gaus = data_blur_poi_gaus * max_signal

# ================================================================================================================
# check the correct of the psf
nr, nc = 2, 3
vmax_gt = data_gt.max() * 0.5
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4 * nc, 2.4 * nr), constrained_layout=True)
axes[0,0].imshow(data_gt[Sx//2],    cmap='gray', vmin=0, vmax=vmax_gt), axes[0,0].set_title('GT (max={:.2f})'.format(data_gt.max()))
axes[0,1].imshow(data_input[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt), axes[0,1].set_title('Blured (max={:.2f})'.format(data_input.max()))
axes[0,2].imshow(data_blur[Sx//2],  cmap='gray', vmin=0, vmax=vmax_gt), axes[0,2].set_title('Blured_c (max={:.2f})'.format(data_blur.max()))
axes[1,2].imshow(np.abs(data_input-data_blur)[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt*0.1)
axes[1,2].set_title('Diff (max={:.2f})'.format(np.abs((data_input-data_blur)).max()))
print(np.max(np.abs(data_input-data_blur)))

axes[1,0].imshow(PSF_align[Sx//2, Sy//2-16 : Sy//2+16, Sz//2-16 : Sz//2+16], cmap='gray', vmin=0, vmax=PSF.max()), axes[1,0].set_title('PSF (xy)')
axes[1,1].imshow(PSF_align[Sx//2-16 : Sx//2+16, Sy//2, Sz//2-16 : Sz//2+16], cmap='gray', vmin=0, vmax=PSF.max()), axes[1,1].set_title('PSF (xz)')
plt.savefig(os.path.join(fig_path, 'psf_check.png'))
