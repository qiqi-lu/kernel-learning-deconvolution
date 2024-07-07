import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva

# ================================================================================================================
dataset_name = 'Microtubule'
# dataset_name = 'Nuclear_Pore_complex'
id_data = 1
fig_path     = os.path.join('outputs', 'figures', dataset_name.lower())
if not os.path.exists(fig_path): os.makedirs(fig_path, exist_ok=True)

# ================================================================================================================
# load data
dataset_path    = os.path.join('F:', os.sep, 'Datasets', 'RCAN3D', 'Confocal_2_STED', dataset_name)
data_gt_path    = os.path.join(dataset_path, 'training', 'gt',  'mt_01.tif')
data_input_path = os.path.join(dataset_path, 'training', 'raw', 'mt_01.tif')

data_gt     = io.imread(data_gt_path).astype(np.float32)
data_input  = io.imread(data_input_path).astype(np.float32)

print('GT: {}, Input: {}'.format(data_gt.shape, data_input.shape))

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
