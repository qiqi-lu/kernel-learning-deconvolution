import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva

dataset_name = 'ZeroShotDeconvNet'
dataset_path = os.path.join('F:', os.sep, 'Datasets', dataset_name,\
    '3D time-lapsing data_LLSM_Mitosis_H2B', 'train_data')

fig_path = os.path.join('outputs', 'figures', dataset_name.lower())
if not os.path.exists(fig_path): os.makedirs(fig_path, exist_ok=True)

with open(os.path.join(dataset_path, 'train.txt')) as f:
    test_txt = f.read().splitlines() 

id_data  = 0
data_raw_path = os.path.join(dataset_path, test_txt[id_data])
data_raw = io.imread(data_raw_path).astype(np.float32)

Sx, Sy, Sz = data_raw.shape

# check the correct of the psf
nr, nc = 2, 3
data_raw = data_raw/data_raw.sum()*100.0*6*1021*1024
print(data_raw.min())
vmax_gt = data_raw.max() * 0.6
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4 * nc, 2.4 * nr), constrained_layout=True)
# [ax.set_axis_off() for ax in axes[0:2,0:2].ravel()]

axes[0,1].imshow(data_raw[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt), axes[0,1].set_title('RAW (sum={:.2f})'.format(data_raw.max()))
axes[0,2].plot(data_raw[Sx//2, 100, 50:500], 'green')

axes[1,1].imshow(data_raw[Sx//2+1], cmap='gray', vmin=0, vmax=vmax_gt), axes[1,1].set_title('RAW (sum={:.2f})'.format(data_raw.max()))
axes[1,2].plot(data_raw[Sx//2+1, 100, 50:500], 'green')

plt.savefig(os.path.join(fig_path, 'data_check.png'))
