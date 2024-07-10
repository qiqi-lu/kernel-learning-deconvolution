import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as skiio
import scipy.io as sciio
import numpy as np
import os
from utils import evaluation as eva

from tabulate import tabulate as tabu

def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")

# ------------------------------------------------------------------------------
def cal_ssim(x, y):
    # need 3D input
    if y.shape[0] >= 7: # the size of the filter in SSIM is at least 7
        return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
            multichannel=False, channle_axis=None, version_wang=False)
    else:
        return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
            multichannel=True, channle_axis=0, version_wang=False)

def cal_psnr(x, y):
    # need 3D input
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())

# ------------------------------------------------------------------------------
dataset_name = ['SimuBeads3D_128', 'SimuMix3D_128']
id_data = [0, 1, 2, 3, 4, 5]
# std_gauss, poisson, ratio  = 0.5, 1, 0.1
# std_gauss, poisson, ratio  = 0.5, 1, 0.3
# std_gauss, poisson, ratio  = 0.5, 1, 1
std_gauss, poisson, ratio  = 0, 0, 1
scale_factor = 1

y_all, y_pred_all  = [], []
psnr_all, ssim_all = [], []

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print('load results ...')
for specimen in dataset_name:
    path_fig_data = os.path.join('outputs', 'figures', specimen,\
        f'scale_{scale_factor}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}')
    print('>> Load result from :', path_fig_data)

    y, y_pred = [], []
    for i in id_data:
        path_sample = os.path.join(path_fig_data, f'sample_{i}')

        # ground truth 
        y.append(skiio.imread(os.path.join(path_sample, 'kernelnet_ss', 'y.tif')))

        # prediction
        y_deconv = []
        y_deconv.append(skiio.imread(os.path.join(path_sample, 'kernelnet_ss',\
            'x.tif'))) # RAW
        y_deconv.append(skiio.imread(os.path.join(path_sample,\
            'wiener_butterworth', 'deconv_2.tif'))) # WB
        y_deconv.append(skiio.imread(os.path.join(path_sample, 'kernelnet_ss',\
            'y_pred_all.tif'))[2]) # our method
        y_deconv.append(skiio.imread(os.path.join(path_sample, 'kernelnet',\
            'y_pred_all.tif'))[2]) # our method
        y_pred.append(y_deconv)
    y_all.append(y)
    y_pred_all.append(y_pred)

y_all, y_pred_all = np.array(y_all), np.array(y_pred_all)

# (specimen, sample, methods, Nz, Ny, Nx)
num_specimen, num_sample, num_method = y_pred_all.shape[0:3]

# ------------------------------------------------------------------------------
psnr = np.zeros(shape=(num_specimen, num_sample, num_method))
ssim = np.zeros_like(psnr)
ncc  = np.zeros_like(psnr)

print('-'*80)
for i in range(num_specimen):
    print('Specimen:', i)
    for j in range(num_sample):
        for k in range(num_method):
            psnr[i,j,k] = cal_psnr(y_pred_all[i,j,k], y_all[i,j])
            ssim[i,j,k] = cal_ssim(y_pred_all[i,j,k], y_all[i,j])
            ncc[i,j,k]  = eva.NCC(y_pred_all[i,j,k], y_all[i,j])
    print('-'*80)
    print(tabulate(psnr[i].transpose()))
    print('-'*80)
    print(tabulate(ssim[i].transpose()))
    print('-'*80)
    print(tabulate(ncc[i].transpose()))
    print('-'*80)

# last iteration
psnr_mean, psnr_std = psnr.mean(axis=1), psnr.std(axis=1)
ssim_mean, ssim_std = ssim.mean(axis=1), ssim.std(axis=1)
ncc_mean, ncc_std   = ncc.mean(axis=1), ncc.std(axis=1)

# ------------------------------------------------------------------------------
print('plot ...')
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
axes = axes.ravel()

width, spacing = 0.1, 0.15
methods_name   = ['RAW', 'WB', 'KLD-ss','KLD']
methods_colors = ['#FFF455', '#FFC700', '#4CCD99','#007F73']

dict_bar = {'width': width, 'capsize': 2.5, 'edgecolor': 'black',\
    'linewidth': 0.5, 'error_kw':{'elinewidth': 0.5, 'capthick': 0.5}}

for ax in axes.ravel():
    for pos in ['top','bottom','left','right']:
        ax.spines[pos].set_linewidth(0.5)
        ax.tick_params(width=0.5)

ind = np.arange(psnr_mean.shape[0])*0.8

for i, mean, std in zip([0, 1, 2],[psnr_mean, ssim_mean, ncc_mean],\
    [psnr_std, ssim_std, ncc_std]):
    axes[i].bar(ind - 2*spacing, mean[:, 3], yerr=std[:, 3],\
        label=methods_name[3], color=methods_colors[3], **dict_bar)
    axes[i].bar(ind - spacing, mean[:, 2], yerr=std[:, 2],\
        label=methods_name[2], color=methods_colors[2], **dict_bar)
    axes[i].bar(ind,  mean[:, 1], yerr=std[:, 1],\
        label=methods_name[1], color=methods_colors[1], **dict_bar)
    axes[i].bar(ind + spacing, mean[:, 0], yerr=std[:, 0],\
        label=methods_name[0], color=methods_colors[0], **dict_bar)

axes[0].set_ylabel('PSNR')
if ratio == 1:   axes[0].set_ylim([21, 34])
if ratio == 0.3: axes[0].set_ylim([20, 34])
if ratio == 0.1: axes[0].set_ylim([20, 30])

axes[1].set_ylabel('SSIM')
if ratio == 1: 
    axes[1].set_ylim([0.65, 1])
    axes[1].set_yticks([0.6, 0.7, 0.8, 0.9, 1],\
        labels=['0.6', '0.7', '0.8', '0.9', '1.0'])
if ratio == 0.3:
    axes[1].set_ylim([0.55, 1.0])
    axes[1].set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1],\
        labels=['0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9',\
        '1.0'])
if ratio == 0.1:
    axes[1].set_ylim([0.45, 0.7])
    axes[1].set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7],\
        labels=['0.45', '0.5', '0.55', '0.6', '0.65', '0.7'])

axes[2].set_ylabel('NCC')
if ratio == 1: 
    axes[2].set_ylim([0.75, 1.0])
    axes[2].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0],\
                labels=['0.75', '0.8', '0.85', '0.9', '0.95', '1.0'])
if ratio == 0.3:
    axes[2].set_ylim([0.75, 1.05])
    axes[2].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0],\
                labels=['0.75', '0.8', '0.85', '0.9', '0.95', '1.0'])
if ratio == 0.1:
    axes[2].set_ylim([0.7, 1.0])
    axes[2].set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],\
                labels=['0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0'])

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Beads', 'Mix'))
axes[0].legend(edgecolor='white', fontsize='small',ncol=1)

plt.savefig(os.path.join('outputs', 'figures',\
    f'metrics_simu_{std_gauss}_poiss_{poisson}_ratio_{ratio}_ss.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join('outputs', 'figures',\
    f'metrics_simu_{std_gauss}_poiss_{poisson}_ratio_{ratio}_ss.svg'))