import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
import numpy as np
import os, torch
from utils import evaluation as eva
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import utils.dataset_utils as utils_data

# -----------------------------------------------------------------------------------
dataset_name_test = 'zeroshotdeconvnet'
fig_path_data   = os.path.join('outputs', 'figures', dataset_name_test)
dataset_path = 'F:\\Datasets\\ZeroShotDeconvNet\\3D time-lapsing data_LLSM_Mitosis_H2B'
with open(os.path.join(dataset_path, 'raw.txt')) as f:
    raw_txt = f.read().splitlines()

# -----------------------------------------------------------------------------------
# load results
print('>> Load result from :', fig_path_data)
methods = ['traditional', 'gaussian', 'butterworth', 'wiener_butterworth','kernelnet']

kernel_fp = io.imread(os.path.join(fig_path_data, 'kernel_fp.tif'))
y = io.imread(os.path.join(fig_path_data, 'sample_0', 'kernelnet', 'y.tif'))

fig_path_sample = os.path.join(fig_path_data, f'sample_0')
kernel_bp = []
for meth in methods[0:-1]:
    kernel_bp.append(io.imread(os.path.join(fig_path_sample, meth, 'deconv_bp.tif')))
kernel_bp.append(io.imread(os.path.join(fig_path_data, 'kernel_bp.tif')))
kernel_bp = np.array(kernel_bp)

Nmeth, Sz, Sx, Sy = kernel_bp.shape
print('Num of methods: {}, image shape: {}'.format(Nmeth, (Sz, Sx, Sy)))

# --------------------------------------------------------------------------------------
# show OTF
# --------------------------------------------------------------------------------------
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3.6 * nc, 3.6 * nr),\
                         constrained_layout=True)

def plot_profile(axes, psf, bp_kernel, s=None, color=None, label=None):
    if s == None: s = psf.shape
    psf_fft = utils_data.fft_n(psf, s=s)
    bp_ker_fft = utils_data.fft_n(bp_kernel, s=s)
    S_ker = bp_kernel.shape
    S_fft =  psf_fft.shape
    a = np.abs(bp_ker_fft)
    b = np.abs(psf_fft*bp_ker_fft)

    axes[0,0].plot(bp_kernel[S_ker[0]//2, S_ker[1]//2, :], color=color, label=label)
    axes[0,1].plot(a[S_fft[0]//2, S_fft[1]//2, S_fft[2]//2:], color=color, label=label)
    axes[0,2].plot(b[S_fft[0]//2, S_fft[1]//2, S_fft[2]//2:], color=color, label=label)

    axes[1,0].plot(bp_kernel[:, S_ker[1]//2, S_ker[2]//2], color=color, label=label)
    axes[1,1].plot(a[S_fft[0]//2:, S_fft[1]//2, S_fft[2]//2], color=color, label=label)
    axes[1,2].plot(b[S_fft[0]//2:, S_fft[1]//2, S_fft[2]//2], color=color, label=label)

axes[0,0].axhline(y=0.0, color='black'), axes[1,0].axhline(y=0.0, color='black')
axes[0,0].set_title('BP (x)'), axes[0,1].set_title('|FT(BP)| (x)'), axes[0,2].set_title('|FT(FP) x FT(BP)| (x)')
axes[1,0].set_title('BP (z)'), axes[1,1].set_title('|FT(BP)| (z)'), axes[1,2].set_title('|FT(FP) x FT(BP)| (z)')

methods_color = ['#D04848', '#007F73', '#4CCD99', '#FFC700', '#FFF455']

plot_profile(axes, psf=kernel_fp, bp_kernel=kernel_bp[0], s=y.shape, color=methods_color[4], label=methods[0])
plot_profile(axes, psf=kernel_fp, bp_kernel=kernel_bp[1], s=y.shape, color=methods_color[3], label=methods[1])
plot_profile(axes, psf=kernel_fp, bp_kernel=kernel_bp[2], s=y.shape, color=methods_color[2], label=methods[2])
plot_profile(axes, psf=kernel_fp, bp_kernel=kernel_bp[3], s=y.shape, color=methods_color[1], label=methods[3])
plot_profile(axes, psf=kernel_fp, bp_kernel=kernel_bp[4], s=y.shape, color=methods_color[0], label=methods[4])

axes[1,2].legend()

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(edgecolor='white',fontsize='x-small')

axes[0,0].set_xlabel('Pixel')
axes[1,0].set_xlabel('Pixel')

for ax in axes[:,1:].ravel():
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Normalized value')
    
plt.savefig(os.path.join(fig_path_data, 'profile_bp_fft.png'))
# -----------------------------------------------------------------------------------