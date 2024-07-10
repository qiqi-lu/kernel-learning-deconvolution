import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import utils.dataset_utils as utils_data
from tabulate import tabulate as tabu

def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")
# ------------------------------------------------------------------------------
dataset_name = 'SimuMix3D_128'
# dataset_name = 'SimuMix3D_256'

std_gauss = [0, 0.5]
poisson   = [0, 1]
ratio     = [1, 0.3]

num_data  = [80, 3]
id_repeat = [1, 1]

path_fig = os.path.join('outputs', 'figures', dataset_name)

# ------------------------------------------------------------------------------
# load kernels
# ------------------------------------------------------------------------------
print('load kernels ...')
ker_BP  = []

# conventional backward kernels
methods = ['traditional', 'wiener_butterworth']
for meth in methods:
    ker_BP.append(io.imread(os.path.join(path_fig,\
        'scale_1_gauss_0_poiss_0_ratio_1', 'sample_6', meth, 'deconv_bp.tif')))
# learned backeard kernels
for i in range(len(std_gauss)):
    path_fig_data = os.path.join('outputs', 'figures', dataset_name,\
        f'scale_1_gauss_{std_gauss[i]}_poiss_{poisson[i]}_ratio_{ratio[i]}',\
        f'kernels_bc_{num_data[i]}_re_{id_repeat[i]}')
    ker_BP.append(io.imread(os.path.join(path_fig_data, 'kernel_bp.tif')))
# true forward kernel
ker_true = io.imread(os.path.join(path_fig_data, 'kernel_true.tif'))

# ------------------------------------------------------------------------------
y = io.imread(os.path.join('outputs', 'figures', dataset_name,\
    f'scale_1_gauss_0_poiss_0_ratio_1',
    f'sample_0', 'kernelnet', 'y.tif'))

s_fft = y.shape
# ------------------------------------------------------------------------------
# show backward kernel image
# ------------------------------------------------------------------------------
print('plot backward kernels ...')
nr, nc = 3, 8
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

def show(ker_fp, ker_bp, axes, s=None, title=''):
    dict_ker = {'cmap': 'hot', 'vmin': 0.0}

    ker_fp_fft  = utils_data.fft_n(ker_fp, s=s)
    ker_bp_fft  = utils_data.fft_n(ker_bp, s=s)
    N_kb = ker_bp.shape
    N_kf_ft = ker_fp_fft.shape
    a = np.abs(ker_bp_fft)
    b = np.abs(ker_fp_fft*ker_bp_fft)

    axes[0,0].imshow(ker_bp[N_kb[0]//2],       vmax=ker_bp.max(), **dict_ker)    
    axes[0,1].imshow(ker_bp[:, N_kb[1]//2, :], vmax=ker_bp.max(), **dict_ker)
    axes[1,0].imshow(a[N_kf_ft[0]//2],         vmax=a.max(), **dict_ker)  
    axes[1,1].imshow(a[:,N_kf_ft[1]//2,:],     vmax=a.max(), **dict_ker)
    axes[2,0].imshow(b[N_kf_ft[0]//2],         vmax=b.max(), **dict_ker)   
    axes[2,1].imshow(b[:,N_kf_ft[1]//2,:],     vmax=b.max(), **dict_ker)

show(ker_true, ker_BP[0], axes=axes[:,0:2], s=s_fft, title='Traditional')
show(ker_true, ker_BP[1], axes=axes[:,2:4], s=s_fft, title='WB')
show(ker_true, ker_BP[2], axes=axes[:,4:6], s=s_fft, title='KLD (NF)')
show(ker_true, ker_BP[3], axes=axes[:,6: ], s=s_fft, title='KLD (N)')

plt.savefig(os.path.join(path_fig, 'kernel_bp.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig, 'kernel_bp.svg'))

# ------------------------------------------------------------------------------
# plot FFT of backward kernels
# ------------------------------------------------------------------------------
print('plot fft of backward kernels ...')
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

def plot_profile(axes, ker_fp, ker_bp, s=None, color=None, label=None):
    dict_ker_profile = {'color': color, 'label': label, 'linewidth': 1}

    ker_fp_fft = utils_data.fft_n(ker_fp, s=s)
    ker_bp_fft = utils_data.fft_n(ker_bp, s=s)
    N_kb    = ker_bp.shape
    N_kf_ft = ker_fp_fft.shape
    a = np.abs(ker_bp_fft)
    b = np.abs(ker_fp_fft*ker_bp_fft)

    line_1 = ker_bp[N_kb[0]//2, N_kb[1]//2, :]
    line_2 = a[N_kf_ft[0]//2, N_kf_ft[1]//2, N_kf_ft[2]//2:]
    line_3 = b[N_kf_ft[0]//2, N_kf_ft[1]//2, N_kf_ft[2]//2:]
    axes[0,0].plot(line_1, **dict_ker_profile)
    axes[0,1].plot(line_2, **dict_ker_profile)
    axes[0,2].plot(line_3, **dict_ker_profile)

    line_4 = ker_bp[:, N_kb[1]//2, N_kb[2]//2]
    line_5 = a[N_kf_ft[0]//2:, N_kf_ft[1]//2, N_kf_ft[2]//2]
    line_6 = b[N_kf_ft[0]//2:, N_kf_ft[1]//2, N_kf_ft[2]//2]
    axes[1,0].plot(line_4, **dict_ker_profile)
    axes[1,1].plot(line_5, **dict_ker_profile)
    axes[1,2].plot(line_6, **dict_ker_profile)

    print(tabulate([line_1.tolist(), line_2.tolist(), line_3.tolist()]))
    print(tabulate([line_4.tolist(), line_5.tolist(), line_6.tolist()]))

axes[0,0].axhline(y=0.0, color='black')
axes[1,0].axhline(y=0.0, color='black')

methods_color = ['black', '#6895D2', '#D04848', '#F3B95F']
methods_name  = ['Traditional', 'WB', 'KLD (NF)', 'KLD (N)',]

for i in range(len(methods_name)):
    print('-'*80)
    print(methods_name[i])
    print('-'*80)
    plot_profile(axes, ker_fp=ker_true, ker_bp=ker_BP[i], s=s_fft,\
        color=methods_color[i], label=methods_name[i])
    print('-'*80)

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
axes[0,0].legend(edgecolor='white',fontsize='x-small')
axes[0,0].set_xlabel('Pixelx'), axes[1,0].set_xlabel('Pixelz')

for ax in axes[:,1:].ravel():
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.set_xlabel('Frequency')
    # ax.set_ylabel('value')
    
plt.savefig(os.path.join(path_fig, 'kernel_bp_fft'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig, 'kernel_bp_fft.svg'))
