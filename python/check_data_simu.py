import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva

# ================================================================================================================
data_set_name_test = 'SimuMix3D_128'
# data_set_name_test = 'SimuBeads3D_128'
id_data = [0,1,2,3,4,5]
snr = []

for i_sample in id_data:
    path_data_nf = os.path.join('outputs', 'figures', data_set_name_test,\
        f'scale_1_noise_0', f'sample_{i_sample}', 'kernelnet')

    path_data_n = os.path.join('outputs', 'figures', data_set_name_test,\
        f'scale_1_noise_0.5', f'sample_{i_sample}', 'kernelnet')

    data_gt       = io.imread(os.path.join(path_data_nf, 'y.tif')).astype(np.float32)
    data_input_nf = io.imread(os.path.join(path_data_nf, 'x.tif')).astype(np.float32)
    data_input_n  = io.imread(os.path.join(path_data_n,  'x.tif')).astype(np.float32)
    snr.append(eva.SNR(data_input_nf, data_input_n))

snr = np.array(snr)
print(snr)
print(snr.mean(), snr.std())

os._exit(0)
# ================================================================================================================
# generate input
Sx, Sy, Sz = data_gt.shape

# ================================================================================================================
# check the correct of the psf
nr, nc = 2, 3
data_gt = data_gt/data_gt.sum()*100.0*6*1021*1024
data_input = data_input/data_input.sum()*100.0*6*1021*1024
print(data_gt.min())
print(data_input.min())

vmax_gt = data_gt.max() * 0.6
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4 * nc, 2.4 * nr), constrained_layout=True)
# [ax.set_axis_off() for ax in axes[0:2,0:2].ravel()]

axes[0,0].imshow(data_gt[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt), axes[0,0].set_title('GT (sum={:.2f})'.format(data_gt.max()))
axes[0,1].imshow(data_input[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt), axes[0,1].set_title('RAW (sum={:.2f})'.format(data_input.max()))
axes[0,2].plot(data_gt[Sx//2, 100, 50:500], 'red')
axes[0,2].plot(data_input[Sx//2, 100, 50:500], 'green')

axes[1,0].imshow(data_gt[Sx//2+1], cmap='gray', vmin=0, vmax=vmax_gt), axes[1,0].set_title('GT (sum={:.2f})'.format(data_gt.max()))
axes[1,1].imshow(data_input[Sx//2+1], cmap='gray', vmin=0, vmax=vmax_gt), axes[1,1].set_title('RAW (sum={:.2f})'.format(data_input.max()))
axes[1,2].plot(data_gt[Sx//2+1, 100, 50:500], 'red')
axes[1,2].plot(data_input[Sx//2+1, 100, 50:500], 'green')

plt.savefig(os.path.join(fig_path, 'data_check.png'))
