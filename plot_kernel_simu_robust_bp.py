import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import utils.dataset_utils as utils_data
import utils.evaluation as eva
from scipy import stats

# ------------------------------------------------------------------------------
dataset_name = 'SimuMix3D_128'
path_fig = os.path.join('outputs', 'figures', dataset_name)

# ------------------------------------------------------------------------------
# rubost to training sample
para_data = [[0, 0, 1], [0.5, 1, 1], [0.5, 1, 0.3], [0.5, 1, 0.1]] # std_gauss, poisson, ratio
num_data  = [1, 2, 3]
id_repeat = [1, 2, 3]

kb = []
for para in para_data:
    path_kernel = os.path.join('outputs', 'figures', dataset_name,\
        f'scale_1_gauss_{para[0]}_poiss_{para[1]}_ratio_{para[2]}')
    tmp = []
    for bc in num_data:
        tmpp = []
        for re in id_repeat:
            tmpp.append(io.imread(os.path.join(path_kernel,\
                f'kernels_bc_{bc}_re_{re}', 'kernel_bp.tif')))
        tmp.append(tmpp)
    kb.append(tmp)
kb = np.array(kb)
# ------------------------------------------------------------------------------
# calculate metric value
N_nl, N_data, N_rep = kb.shape[0:3]
print(kb.shape) # dataset, num of train data, num of repeat

pearson = np.zeros(shape=(N_nl, N_data, N_rep))
ratio = [1, 1, 0.3, 0.1]
for i in range(N_nl):
    for j in range(N_data):
        pearson[i,j,0] = stats.pearsonr(x=kb[i,j,0].flatten(),\
            y=kb[i,j,1].flatten())[0]
        pearson[i,j,1] = stats.pearsonr(x=kb[i,j,0].flatten(),\
            y=kb[i,j,2].flatten())[0]
        pearson[i,j,2] = stats.pearsonr(x=kb[i,j,1].flatten(),\
            y=kb[i,j,2].flatten())[0]
print(pearson)

pearson_mean = pearson.mean(axis=-1)
pearson_std  = pearson.std(axis=-1)
print('mean:', pearson_mean)
# ------------------------------------------------------------------------------
nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

dict_line = {'linewidth': 0.5, 'capsize':2, 'elinewidth':0.5, 'capthick':0.5}

noise_level = ['NF', '20', '15', '10']
colors = ['black', 'red', 'green', 'blue']

for i in range(N_nl):
    axes[0].errorbar(x=num_data, y=pearson_mean[i], yerr=pearson_std[i],\
        color=colors[0], **dict_line)
    axes[0].plot(num_data, pearson_mean[i],'.', color=colors[i],\
        label='SNR='+noise_level[i])
axes[0].legend(edgecolor='white', fontsize='x-small')
axes[0].set_ylabel('PCC')
axes[0].set_ylim([0.94, 1])

for ax in axes.ravel():
    ax.set_xticks(ticks=num_data, labels=num_data)
    ax.set_xlabel('Number of samples')

plt.savefig(os.path.join(path_fig, 'kb.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig, 'kb.svg'))