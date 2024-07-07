import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
import numpy as np
import os
from tabulate import tabulate as tabu

# ------------------------------------------------------------------------------
def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")

def save2txt(file, arr):
    with open(file, 'w') as outfile:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        nslice = arr.shape[0]
        for i in range(nslice):
            np.savetxt(outfile, arr[i])

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
# dataset_name = ['Microtubule', 'Nuclear_Pore_complex']
dataset_name = ['Microtubule2', 'Nuclear_Pore_complex2']
name_net = 'kernelnet'
id_data  = [0, 1, 2, 3, 4, 5]

# ------------------------------------------------------------------------------
y_all, y_test_all = [], []

for specimen in dataset_name:
    fig_path_data = os.path.join('outputs', 'figures', specimen,\
        'scale_1_gauss_0_poiss_0_ratio_1')
    print('load result from :', fig_path_data)

    y, y_test = [], []
    for i in id_data:
        tmp = []
        fig_path_sample = os.path.join(fig_path_data, f'sample_{i}')
        y.append(io.imread(os.path.join(fig_path_sample, name_net, 'y.tif')))
        tmp.append(io.imread(os.path.join(fig_path_sample, name_net, 'x.tif')))
        tmp.append(io.imread(os.path.join(fig_path_sample, 'deconvblind',\
            'deconv.tif')))

        if dataset_name[0] == 'Microtubule':
            tmp.append(io.imread(os.path.join(fig_path_sample, 'traditional',\
                'deconv.tif'))) # use learned PSF
        if dataset_name[0] == 'Microtubule2':
            tmp.append(io.imread(os.path.join(fig_path_sample, 'traditional',\
                'deconv_20.tif'))) # use learned PSF

        tmp.append(io.imread(os.path.join(fig_path_sample, name_net,\
            'y_pred_all.tif'))[-1])
        y_test.append(tmp)
    y_all.append(y)
    y_test_all.append(y_test)

y_all = np.array(y_all)
y_test_all = np.transpose(np.array(y_test_all), axes=(0, 2, 1, 3, 4, 5))

num_specimen, num_meth, num_sample = y_test_all.shape[0:3]
print('Num of specimen:', num_specimen, 'Num of method:', num_meth,\
      'Num of sample:', num_sample)

# ------------------------------------------------------------------------------
psnr = np.zeros(shape=(num_specimen, num_meth, num_sample))
ssim = np.zeros_like(psnr)
ncc  = np.zeros_like(psnr)

for i in range(num_specimen):
    print('-'*80)
    print('specimen :', i)
    print('-'*80)
    for j in range(num_meth):
        for k in range(num_sample):
            psnr[i,j,k] = cal_psnr(y_test_all[i,j,k], y_all[i,k])
            ssim[i,j,k] = cal_ssim(y_test_all[i,j,k], y_all[i,k])
            ncc[i,j,k]  = eva.NCC(y_test_all[i,j,k], y_all[i,k])
    print('-'*80)
    print(tabulate(psnr[i].tolist()))
    print('-'*80)
    print(tabulate(ssim[i].tolist()))
    print('-'*80)
    print(tabulate(ncc[i].tolist()))
    print('-'*80)
# ------------------------------------------------------------------------------
nr, nc = 3, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

psnr_mean, psnr_std = psnr.mean(axis=-1), psnr.std(axis=-1)
ssim_mean, ssim_std = ssim.mean(axis=-1), ssim.std(axis=-1)
ncc_mean, ncc_std   = ncc.mean(axis=-1), ncc.std(axis=-1)

print('-'*80)
print('PSNR (mean)')
print(tabulate(psnr_mean.tolist()))
print('SSIM (mean)')
print(tabulate(ssim_mean.tolist()))
print('NCC  (mean)')
print(tabulate(ncc_mean.tolist()))
print('-'*80)

width, spacing = 0.1, 0.15
method_name   = ['RAW', 'DeconvBlind', 'RLD# (200)', 'KLD']
method_colors = ['#6895D2', '#F3B95F', '#FDE767', '#D04848']
ind = np.arange(psnr_mean.shape[0])*0.8
start = ind - spacing*(num_meth - 1)/2

# axes[0].set_ylabel('PSNR'), axes[0].set_ylim([20, 26])
axes[0].set_ylabel('PSNR'), axes[0].set_ylim([20, 26.5])
axes[1].set_ylabel('SSIM'), axes[1].set_ylim([0.2, 0.5])
axes[2].set_ylabel('NCC'),  axes[2].set_ylim([0.6, 0.8])

dict_bar = {'capsize':2.5, 'error_kw': {'elinewidth':0.5, 'capthick':0.5}}

for i in range(num_meth):
    id_meth = num_meth-i-1

    axes[0].bar(start + spacing*i , psnr_mean[:, id_meth], width,\
        yerr=psnr_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

    axes[1].bar(start + spacing*i , ssim_mean[:, id_meth], width,\
        yerr=ssim_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

    axes[2].bar(start + spacing*i , ncc_mean[:, id_meth], width,\
        yerr=ncc_std[:, id_meth],\
        label=method_name[id_meth], color=method_colors[id_meth], **dict_bar)

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(ind)
    ax.set_xticklabels(('MT', 'NPC'))
axes[0].legend(edgecolor='white', fontsize='x-small')

if dataset_name[0] == 'Microtubule':
    plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_3d.png'))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_3d.svg'))

if dataset_name[0] == 'Microtubule2':
    plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_3d_2.png'))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(os.path.join('outputs', 'figures', 'metrics_real_3d_2.svg'))