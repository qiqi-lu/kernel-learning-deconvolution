import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
import numpy as np
import os
from utils import evaluation as eva
from tabulate import tabulate as tabu

# ------------------------------------------------------------------------------
def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")

# ------------------------------------------------------------------------------
def cal_ssim(x, y):
    return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
        version_wang=False)
def cal_psnr(x, y):
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())

# ------------------------------------------------------------------------------
dataset_name = ['F-actin_Nonlinear', 'Microtubules2']
# ------------------------------------------------------------------------------
name_net = 'kernelnet'
std_gauss, scale_factor = 9, 1
id_data = [0, 1, 2, 3, 4, 5]
# id_data = [6, 7, 8, 9, 10]

# ------------------------------------------------------------------------------
y_all, y_test_all = [], []
print('-'*80)
for specimen in dataset_name:
    path_result = os.path.join('outputs', 'figures', specimen,\
        f'scale_{scale_factor}_gauss_{std_gauss}_poiss_1_ratio_1')
    print('load result from :', path_result)

    y, y_test = [], []
    for i in id_data:
        tmp = []
        path = os.path.join(path_result, f'sample_{i}')
        y.append(  io.imread(os.path.join(path, name_net, 'y.tif')))
        tmp.append(io.imread(os.path.join(path, name_net, 'x.tif')))
        tmp.append(io.imread(os.path.join(path, 'deconvblind', 'deconv.tif')))
        tmp.append(io.imread(os.path.join(path, 'traditional', 'deconv.tif')))
        y_kld = io.imread(os.path.join(path, name_net, 'y_pred_all.tif'))
        # special processing
        if y_kld.shape[-1] in [3, 4]:
            y_kld = np.transpose(y_kld,(-1, 0, 1))
        tmp.append(y_kld[2])

        y_test.append(tmp)
    y_all.append(y)
    y_test_all.append(y_test)

# (2, 5, 502, 502)
y_all = np.array(y_all)
# (2, 4, 5, 502, 502)
y_test_all = np.transpose(np.array(y_test_all), axes=(0, 2, 1, 3, 4))

num_specimen, num_meth, num_sample = y_test_all.shape[0:3]
print('Num of specimen:', num_specimen, 'Num of method:', num_meth,\
      'Num of sample:', num_sample)

# ------------------------------------------------------------------------------
psnr = np.zeros(shape=(num_specimen, num_meth, num_sample))
ssim = np.zeros_like(psnr)
ncc  = np.zeros_like(psnr)

for i in range(num_specimen):
    print('-'*80)
    print('specimen:', i)
    for j in range(num_meth):
        for k in range(num_sample):
            psnr[i,j,k] = cal_psnr(y_test_all[i,j,k], y_all[i,k])
            ssim[i,j,k] = cal_ssim(y_test_all[i,j,k], y_all[i,k])
            ncc[i,j,k]  = eva.NCC( y_test_all[i,j,k], y_all[i,k])
    print('-'*80)
    print(tabulate(psnr[i]))
    print('-'*80)
    print(tabulate(ssim[i]))
    print('-'*80)
    print(tabulate(ncc[i]))
    print('-'*80)

psnr_mean, psnr_std = psnr.mean(axis=-1), psnr.std(axis=-1)
ssim_mean, ssim_std = ssim.mean(axis=-1), ssim.std(axis=-1)
ncc_mean, ncc_std   = ncc.mean(axis=-1), ncc.std(axis=-1)

print('-'*80)
print('PSNR: ', psnr_mean)
print('SSIM:',  ssim_mean)
print('NCC:',   ncc_mean)
print('-'*80)

method_name   = ['RAW', 'DeconvBlind', 'RLD# (200)', 'KLD']
method_colors = ['#6895D2', '#F3B95F', '#FDE767', '#D04848']
# ------------------------------------------------------------------------------
nr, nc = 3, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

width, spacing = 0.1, 0.15

ind = np.arange(psnr_mean.shape[0])*0.8
start = ind - spacing*(num_meth - 1)/2

axes[0].set_ylabel('PSNR'), axes[0].set_ylim([25, 36.5])
axes[1].set_ylabel('SSIM'), axes[1].set_ylim([0.4, 1.0])
axes[2].set_ylabel('NCC'),  axes[2].set_ylim([0.75, 1.0])

dict_bar = {'capsize':2.5, 'error_kw': {'elinewidth':0.5, 'capthick':0.5}}

for i in range(num_meth):
    id_meth = num_meth-i-1

    axes[0].bar(start + spacing*i ,\
        psnr_mean[:, id_meth], width, yerr=psnr_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

    axes[1].bar(start + spacing*i ,\
        ssim_mean[:, id_meth], width, yerr=ssim_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

    axes[2].bar(start + spacing*i ,\
        ncc_mean[:, id_meth], width, yerr=ncc_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

xticks = ['F-actin', 'MT']
axes[0].legend(edgecolor='white', fontsize='x-small')
for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(ind)
    ax.set_xticklabels(xticks)

plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_2d.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_2d.svg'))