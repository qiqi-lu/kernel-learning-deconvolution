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
std_gauss, poisson, ratio  = 0.5, 1, 0.1
# std_gauss, poisson, ratio  = 0.5, 1, 0.3
# std_gauss, poisson, ratio  = 0.5, 1, 1
# std_gauss, poisson, ratio  = 0, 0, 1
scale_factor = 1

# suffix_net = '_ss'
suffix_net = ''
# ------------------------------------------------------------------------------
y_all, y_pred_all     = [], []
y_deconv_metrics_all  = []
psnr_all, ssim_all    = [], []
res_xy_all, res_z_all = [], []

net_name = f'kernelnet{suffix_net}'
methods  = ['traditional', 'gaussian', 'butterworth', 'wiener_butterworth']

# ------------------------------------------------------------------------------
for specimen in dataset_name:
    path_fig_data = os.path.join('outputs', 'figures', specimen,\
        f'scale_{scale_factor}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}')
    print('>> Load result from : ', path_fig_data)

    y, y_pred = [], []
    y_deconv_metrics = []
    res_xy, res_z = [], []

    for i in id_data:
        path_sample = os.path.join(path_fig_data, f'sample_{i}')

        # ground truth 
        y.append(skiio.imread(os.path.join(path_sample, net_name, 'y.tif')))

        # prediction
        y_deconv = []
        y_deconv.append(skiio.imread(os.path.join(path_sample, net_name,\
            'x.tif'))) # RAW
        for meths in methods[:-1]: # conventional methods
            y_deconv.append(skiio.imread(os.path.join(path_sample, meths,\
                'deconv_30.tif')))
        y_deconv.append(skiio.imread(os.path.join(path_sample, methods[-1],\
            'deconv_2.tif'))) # WB
        y_deconv.append(skiio.imread(os.path.join(path_sample, net_name,\
            'y_pred_all.tif'))[2]) # our method
        y_pred.append(y_deconv)

        metrics = []
        metrics.append(np.load(os.path.join(path_sample, 'traditional',\
            'deconv_metrics_30.npy')))
        metrics.append(np.load(os.path.join(path_sample, 'gaussian',\
            'deconv_metrics_30.npy')))
        metrics.append(np.load(os.path.join(path_sample, 'butterworth',\
            'deconv_metrics_30.npy')))
        y_deconv_metrics.append(metrics)

        # resolution calculated using matlab
        res_xy.append(sciio.loadmat(os.path.join(path_sample,\
            'res_xy.mat'))['kcMax_xy_all'])
        res_z.append(sciio.loadmat(os.path.join(path_sample,\
            'res_z.mat'))['kcMax_z_all'])
        
    y_deconv_metrics_all.append(y_deconv_metrics)
    y_all.append(y)
    y_pred_all.append(y_pred)
    res_xy_all.append(res_xy)
    res_z_all.append(res_z)
# ------------------------------------------------------------------------------

# (specimen, sample, methods, Nz, Ny, Nx)
y_all, y_pred_all    = np.array(y_all), np.array(y_pred_all)

# (specimen, sample, methods, iter, metrics)
y_deconv_metrics_all = np.array(y_deconv_metrics_all)

# (specimen, sample, slice, methods)
res_xy_all, res_z_all = np.array(res_xy_all), np.array(res_z_all)

# ------------------------------------------------------------------------------
num_specimen, num_sample, num_method = y_pred_all.shape[0:3]
psnr = np.zeros(shape=(num_specimen, num_sample, num_method))
ssim = np.zeros_like(psnr)
ncc  = np.zeros_like(psnr)

for i in range(num_specimen):
    print('Specimen:', i)
    for j in range(num_sample):
        for k in range(num_method):
            psnr[i,j,k] = cal_psnr(y_pred_all[i,j,k], y_all[i,j])
            ssim[i,j,k] = cal_ssim(y_pred_all[i,j,k], y_all[i,j])
            ncc[i,j,k]  = eva.NCC(y_pred_all[i,j,k], y_all[i,j])
    print('-'*80)
    print('[method, sample] (iteration=30)')
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

# the second iteration
psnr_mean_2 = y_deconv_metrics_all[...,2,0].mean(axis=1)
psnr_std_2  = y_deconv_metrics_all[...,2,0].std(axis=1)
ssim_mean_2 = y_deconv_metrics_all[...,2,1].mean(axis=1)
ssim_std_2  = y_deconv_metrics_all[...,2,1].std(axis=1)
ncc_mean_2  = y_deconv_metrics_all[...,2,2].mean(axis=1)
ncc_std_2   = y_deconv_metrics_all[...,2,2].std(axis=1)

for i in range(num_specimen):
    print('Specimen:', i)
    print('-'*80)
    print('[method, sample] (2 iterations)')
    print(tabulate(y_deconv_metrics_all[...,2, 0][i].transpose()))
    print('-'*80)
    print(tabulate(y_deconv_metrics_all[...,2, 1][i].transpose()))
    print('-'*80)
    print(tabulate(y_deconv_metrics_all[...,2, 2][i].transpose()))
    print('-'*80)

# ------------------------------------------------------------------------------
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
axes = axes.ravel()

width, spacing = 0.1, 0.15
# methods_colors = ['red', 'blue', 'green', 'cyan', 'orange','pink']
methods_name = ['KLD', 'WB', 'Butterworth', 'Gaussian', 'Traditional',\
    'RAW']
methods_colors = ['#D04848', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D',\
    '#6895D2']
ind = np.arange(psnr_mean.shape[0])*1.1

dict_bar_30 = {'width': width, 'capsize': 2.5, 'edgecolor': 'black',\
    'linewidth': 0.5, 'error_kw':{'elinewidth': 0.5, 'capthick': 0.5}}
dict_bar_2  = {'width': width, 'capsize': 2.5, 'edgecolor': 'gray',\
    'ecolor': 'gray', 'linewidth': 0.5,\
    'error_kw':{'elinewidth': 0.5, 'capthick': 0.5}}

for ax in axes.ravel():
    for pos in ['top','bottom','left','right']:
        ax.spines[pos].set_linewidth(0.5)
        ax.tick_params(width=0.5)

axes[0].set_ylabel('PSNR')
axes[0].bar(ind - spacing/2 - 2*spacing, psnr_mean[:, 5], yerr=psnr_std[:, 5],\
    label=methods_name[0] + ' (2)', color=methods_colors[0], **dict_bar_30)
axes[0].bar(ind - spacing/2 - spacing,   psnr_mean[:, 4], yerr=psnr_std[:, 4],\
    label=methods_name[1] + ' (2)', color=methods_colors[1], **dict_bar_30)
axes[0].bar(ind - spacing/2,             psnr_mean[:, 3], yerr=psnr_std[:, 3],\
    label=methods_name[2] + ' (30)', color=methods_colors[2], **dict_bar_2)
axes[0].bar(ind - spacing/2,             psnr_mean_2[:, 2], yerr=psnr_std_2[:,2],\
    label=methods_name[2] + ' (2)', color=methods_colors[2], **dict_bar_30)
axes[0].bar(ind + spacing/2 ,            psnr_mean[:, 2], yerr=psnr_std[:, 2],\
    label=methods_name[3] + ' (30)', color=methods_colors[3], **dict_bar_2)
axes[0].bar(ind + spacing/2 ,            psnr_mean_2[:, 1],\
    yerr=psnr_std_2[:,1], label=methods_name[3] + ' (2)',\
    color=methods_colors[3], **dict_bar_30)
axes[0].bar(ind + spacing/2 + spacing ,  psnr_mean[:, 1],\
    yerr=psnr_std[:, 1], label=methods_name[4] + ' (30)',\
    color=methods_colors[4], **dict_bar_2)
axes[0].bar(ind + spacing/2 + spacing ,  psnr_mean_2[:, 0],\
    yerr=psnr_std_2[:,0], label=methods_name[4] + ' (2)',\
    color=methods_colors[4], **dict_bar_30)
axes[0].bar(ind + spacing/2 + 2*spacing, psnr_mean[:, 0],\
    yerr=psnr_std[:, 0], label=methods_name[5],\
    color=methods_colors[5], **dict_bar_30)

if ratio == 1:   axes[0].set_ylim([21, 34])
if ratio == 0.3: axes[0].set_ylim([20, 34])
if ratio == 0.1: axes[0].set_ylim([20, 30])

axes[1].set_ylabel('SSIM')
axes[1].bar(ind - spacing/2 - 2*spacing, ssim_mean[:, 5],\
    yerr=ssim_std[:, 5], label=methods_name[0] + ' (2)',\
    color=methods_colors[0], **dict_bar_30)
axes[1].bar(ind - spacing/2 - spacing,   ssim_mean[:, 4],\
    yerr=ssim_std[:, 4], label=methods_name[1] + ' (2)',\
    color=methods_colors[1], **dict_bar_30)
axes[1].bar(ind - spacing/2,             ssim_mean[:, 3],\
    yerr=ssim_std[:, 3], label=methods_name[2] + ' (30)',\
    color=methods_colors[2], **dict_bar_2)
axes[1].bar(ind - spacing/2,             ssim_mean_2[:, 2],\
    yerr=ssim_std_2[:, 2], label=methods_name[2] + ' (2)',\
    color=methods_colors[2], **dict_bar_30)
axes[1].bar(ind + spacing/2 ,            ssim_mean[:, 2],\
    yerr=ssim_std[:, 2], label=methods_name[3] + ' (30)',\
    color=methods_colors[3], **dict_bar_2)
axes[1].bar(ind + spacing/2 ,            ssim_mean_2[:, 1],\
    yerr=ssim_std_2[:, 1], label=methods_name[3] + ' (2)',\
    color=methods_colors[3], **dict_bar_30)
axes[1].bar(ind + spacing/2 + spacing ,  ssim_mean[:, 1],\
    yerr=ssim_std[:, 1], label=methods_name[4] + ' (30)',\
    color=methods_colors[4], **dict_bar_2)
axes[1].bar(ind + spacing/2 + spacing ,  ssim_mean_2[:, 0],\
    yerr=ssim_std_2[:, 0], label=methods_name[4] + ' (2)',\
    color=methods_colors[4], **dict_bar_30)
axes[1].bar(ind + spacing/2 + 2*spacing, ssim_mean[:, 0],\
    yerr=ssim_std[:, 0], label=methods_name[5],\
    color=methods_colors[5], **dict_bar_30)
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
axes[2].bar(ind - spacing/2 - 2*spacing, ncc_mean[:, 5],\
    yerr=ncc_std[:, 5], label=methods_name[0] + ' (2)',\
    color=methods_colors[0], **dict_bar_30)
axes[2].bar(ind - spacing/2 - spacing,   ncc_mean[:, 4],\
    yerr=ncc_std[:, 4], label=methods_name[1] + ' (2)',\
    color=methods_colors[1], **dict_bar_30)
axes[2].bar(ind - spacing/2,             ncc_mean[:, 3],\
    yerr=ncc_std[:, 3], label=methods_name[2] + ' (30)',\
    color=methods_colors[2], **dict_bar_2)
axes[2].bar(ind - spacing/2,             ncc_mean_2[:, 2],\
    yerr=ncc_std_2[:, 2], label=methods_name[2] + ' (2)',\
    color=methods_colors[2], **dict_bar_30)
axes[2].bar(ind + spacing/2 ,            ncc_mean[:, 2],\
    yerr=ncc_std[:, 2], label=methods_name[3] + ' (30)',\
    color=methods_colors[3], **dict_bar_2)
axes[2].bar(ind + spacing/2 ,            ncc_mean_2[:, 1],\
    yerr=ncc_std_2[:, 1], label=methods_name[3] + ' (2)',\
    color=methods_colors[3], **dict_bar_30)
axes[2].bar(ind + spacing/2 + spacing ,  ncc_mean[:, 1],\
    yerr=ncc_std[:, 1], label=methods_name[4] + ' (30)',\
    color=methods_colors[4], **dict_bar_2)
axes[2].bar(ind + spacing/2 + spacing ,  ncc_mean_2[:, 0],\
    yerr=ncc_std_2[:, 0], label=methods_name[4] + ' (2)',\
    color=methods_colors[4], **dict_bar_30)
axes[2].bar(ind + spacing/2 + 2*spacing, ncc_mean[:, 0],\
    yerr=ncc_std[:, 0], label=methods_name[5],\
    color=methods_colors[5], **dict_bar_30)
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

# # ------------------------------------------------------------------------------
# # resolution
# pixel_size_xy, pixel_size_z = 1, 1
# res_xy_all = 2*pixel_size_xy/res_xy_all
# res_z_all  = 2*pixel_size_z/res_z_all
# resolution_xy = res_xy_all.mean(axis=2)
# resolution_z  = res_z_all.mean(axis=2)

# resolution_xy_mean = resolution_xy.mean(axis=1)
# resolution_xy_std  = resolution_xy.std(axis=1)
# resolution_z_mean  = resolution_z.mean(axis=1)
# resolution_z_std   = resolution_z.std(axis=1)

# resolution_xy_mean_2 = resolution_xy.mean(axis=1)
# resolution_xy_std_2  = resolution_xy.std(axis=1)
# resolution_z_mean_2  = resolution_z.mean(axis=1)
# resolution_z_std_2   = resolution_z.std(axis=1)
# # ------------------------------------------------------------------------------

# axes[3].set_ylabel('Resolution (xy)')
# axes[3].bar(ind - spacing/2 - 2*spacing, resolution_xy_mean[:, 5],\
#     yerr=resolution_xy_std[:, 5], label=methods_name[0] + ' (2)',\
#         color=methods_colors[0], **dict_bar_30)
# axes[3].bar(ind - spacing/2 - spacing,   resolution_xy_mean[:, 4],\
#     yerr=resolution_xy_std[:, 4], label=methods_name[1] + ' (2)',\
#         color=methods_colors[1], **dict_bar_30)
# axes[3].bar(ind - spacing/2,             resolution_xy_mean[:, 3],\
#     yerr=resolution_xy_std[:, 3], label=methods_name[2] + ' (30)',\
#         color=methods_colors[2], **dict_bar_2)
# # axes[3].bar(ind - spacing/2,             resolution_xy_mean_2[:, 2],\
# #     yerr=resolution_xy_std_2[:,2], label=methods_name[2] + ' (2)',\
# #     color=methods_colors[2], **dict_bar_30)
# axes[3].bar(ind + spacing/2 ,            resolution_xy_mean[:, 2],\
#     yerr=resolution_xy_std[:, 2], label=methods_name[3] + ' (30)',\
#         color=methods_colors[3], **dict_bar_2)
# # axes[3].bar(ind + spacing/2 ,            resolution_xy_mean_2[:, 1],\
# #     yerr=resolution_xy_std_2[:,1], label=methods_name[3] + ' (2)',\
# #     color=methods_colors[3], **dict_bar_30)
# axes[3].bar(ind + spacing/2 + spacing ,  resolution_xy_mean[:, 1],\
#     yerr=resolution_xy_std[:, 1], label=methods_name[4] + ' (30)',\
#     color=methods_colors[4], **dict_bar_2)
# # axes[3].bar(ind + spacing/2 + spacing ,  resolution_xy_mean_2[:, 0], \
# #     yerr=resolution_xy_std_2[:,0], label=methods_name[4] + ' (2)',\
# #     color=methods_colors[4], **dict_bar_30)
# axes[3].bar(ind + spacing/2 + 2*spacing, resolution_xy_mean[:, 0],\
#     yerr=resolution_xy_std[:, 0], label=methods_name[5],\
#     color=methods_colors[5], **dict_bar_30)
# axes[3].set_ylim([0, 9])


# axes[4].set_ylabel('Resolution (z)')
# axes[4].bar(ind - spacing/2 - 2*spacing, resolution_z_mean[:, 5],\
#     yerr=resolution_z_std[:, 5], label=methods_name[0] + ' (2)',\
#     color=methods_colors[0], **dict_bar_30)
# axes[4].bar(ind - spacing/2 - spacing,   resolution_z_mean[:, 4],\
#     yerr=resolution_z_std[:, 4], label=methods_name[1] + ' (2)',\
#     color=methods_colors[1], **dict_bar_30)
# axes[4].bar(ind - spacing/2,             resolution_z_mean[:, 3],\
#     yerr=resolution_z_std[:, 3], label=methods_name[2] + ' (30)',\
#     color=methods_colors[2], **dict_bar_2)
# # axes[4].bar(ind - spacing/2,             resolution_xy_mean_2[:, 2],\
# #     yerr=resolution_z_std_2[:,2], label=methods_name[2] + ' (2)',\
# #     color=methods_colors[2], **dict_bar_30)
# axes[4].bar(ind + spacing/2 ,            resolution_z_mean[:, 2],\
#     yerr=resolution_z_std[:, 2], label=methods_name[3] + ' (30)',\
#     color=methods_colors[3], **dict_bar_2)
# # axes[3].bar(ind + spacing/2 ,            resolution_xy_mean_2[:, 1],\
# #     yerr=resolution_z_std_2[:,1], label=methods_name[3] + ' (2)',\
# #     color=methods_colors[3], **dict_bar_30)
# axes[4].bar(ind + spacing/2 + spacing ,  resolution_z_mean[:, 1],\
#     yerr=resolution_z_std[:, 1], label=methods_name[4] + ' (30)',\
#     color=methods_colors[4], **dict_bar_2)
# # axes[3].bar(ind + spacing/2 + spacing ,  resolution_xy_mean_2[:, 0],\
# #     yerr=resolution_z_std_2[:,0], label=methods_name[4] + ' (2)',\
# #     color=methods_colors[4], **dict_bar_30)
# axes[4].bar(ind + spacing/2 + 2*spacing, resolution_z_mean[:, 0],\
#     yerr=resolution_z_std[:, 0], label=methods_name[5],\
#     color=methods_colors[5], **dict_bar_30)
# axes[4].set_ylim([0, 18])

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Beads', 'Mix'))
axes[0].legend(edgecolor='white', fontsize='xx-small',ncol=1)

# axes[5].set_axis_off()

plt.savefig(os.path.join('outputs', 'figures',\
    f'metrics_simu_{std_gauss}_poiss_{poisson}_ratio_{ratio}{suffix_net}.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join('outputs', 'figures',\
    f'metrics_simu_{std_gauss}_poiss_{poisson}_ratio_{ratio}{suffix_net}.svg'))
# ------------------------------------------------------------------------------