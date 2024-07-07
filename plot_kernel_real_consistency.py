import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, torch
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------------
name_net = 'KernelNet'
id_data = 1

fig_path_data_mt  = os.path.join('outputs', 'figures', 'Microtubule', 'scale_1_noise_0')
fig_path_data_npc = os.path.join('outputs', 'figures', 'Nuclear_Pore_complex', 'scale_1_noise_0')
fig_path  = os.path.join('outputs', 'figures')

# load results
def imread(path, name): 
    im = io.imread(os.path.join(path, name))
    if im.shape[-1] == 3:
        im = np.transpose(im, axes=(2, 0, 1))
    return im

ker_FP, ker_BP = [], []
ker_init = imread(fig_path_data_mt, 'kernel_init.tif')
PSF_true = imread(fig_path_data_mt, 'kernel_true.tif')

ker_FP.append(imread(fig_path_data_mt, 'kernel_fp.tif'))
ker_FP.append(imread(fig_path_data_npc, 'kernel_fp.tif'))

ker_BP.append(imread(fig_path_data_mt, 'kernel_bp.tif'))
ker_BP.append(imread(fig_path_data_npc, 'kernel_bp.tif'))

Sx_t, Sy_t, Sz_t = PSF_true.shape
Sx_f, Sy_f, Sz_f = ker_FP[0].shape
Sx_b, Sy_b, Sz_b = ker_BP[0].shape

vmax_psf, color_map_psf = 0.01, 'hot'

x_mt  = imread(os.path.join(fig_path_data_mt, f'sample_{id_data}', 'kernelnet'), 'x.tif')
y_mt  = imread(os.path.join(fig_path_data_mt, f'sample_{id_data}', 'kernelnet'), 'y.tif')

x_npc = imread(os.path.join(fig_path_data_npc, f'sample_{id_data}', 'kernelnet'), 'x.tif')
y_npc = imread(os.path.join(fig_path_data_npc, f'sample_{id_data}', 'kernelnet'), 'y.tif')

Sx, Sy, Sz = y_mt.shape

# -----------------------------------------------------------------------------------
# FP kernel
# -----------------------------------------------------------------------------------
nr, nc = 2, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3.0 * nc, 3.0 * nr),\
    constrained_layout=True)

# axes[0].plot(PSF_true[Sx_t//2, :, Sz_t//2], color='black', label='True')
colors = ['#D04848', '#F3B95F', '#6895D2']
axes[0].plot(ker_init[Sx_t//2, :, Sz_t//2], color=colors[2], label='init')
axes[0].plot(ker_FP[0][Sx_f//2, :, Sz_f//2], color=colors[0], label=name_net+'(NF)')
axes[0].plot(ker_FP[1][Sx_f//2, :, Sz_f//2], color=colors[1], label=name_net+'(N)')

# axes[1].plot(PSF_true[:, Sy_t//2, Sz_t//2], color='black', label='True')
axes[1].plot(ker_init[:, Sy_t//2, Sz_t//2], color=colors[2], label='init')
axes[1].plot(ker_FP[0][:, Sy_f//2, Sz_f//2], color=colors[1], label=name_net+' (NF)')
axes[1].plot(ker_FP[1][:, Sy_f//2, Sz_f//2], color=colors[0], label=name_net+' (N)')

for ax in axes.ravel():
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(edgecolor='white',fontsize='x-small')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.set_xlabel('Pixel')
axes[0].set_title('PSF profile (xy)')
axes[1].set_title('PSF profile (xz)')

plt.savefig(os.path.join(fig_path, 'kernel_fp_real_profile'))
# -----------------------------------------------------------------------------------

nr, nc = 2, 2.4
fig = plt.figure(dpi=300, figsize=(3.0 * nc, 3.0 * nr), layout=None)
gs  = GridSpec(nrows=10, ncols=12, figure=fig)
ax1_1, ax1_2, ax1_3 = fig.add_subplot(gs[0:4, 0:4]), fig.add_subplot(gs[0:4, 4:8]), fig.add_subplot(gs[0:4, 8:12])
ax2_1, ax2_2, ax2_3 = fig.add_subplot(gs[4, 0:4]), fig.add_subplot(gs[4, 4:8]), fig.add_subplot(gs[4, 8:12])
ax3_1, ax3_2, ax3_3 = fig.add_subplot(gs[5:9, 0:4]), fig.add_subplot(gs[5:9, 4:8]), fig.add_subplot(gs[5:9, 8:12])
ax4_1, ax4_2, ax4_3 = fig.add_subplot(gs[9, 0:4]), fig.add_subplot(gs[9, 4:8]), fig.add_subplot(gs[9, 8:12])
for ax in [ax1_1, ax1_2, ax1_3, ax2_1, ax2_2, ax2_3,\
           ax3_1, ax3_2, ax3_3, ax4_1, ax4_2, ax4_3]:
           ax.set_axis_off()

color_map_img = 'gray'
vmax_img_mt  = y_mt.max()*0.6
vmax_img_npc = y_npc.max()*0.6

def interp(x):
    z_scale = x.shape[1]//x.shape[0]//5
    x = torch.tensor(x)[None, None]
    x = torch.nn.functional.interpolate(x, scale_factor=(z_scale, 1, 1), mode='nearest')
    x = x.numpy()[0,0]
    return x

y_mt, x_mt = interp(y_mt), interp(x_mt)
y_npc, x_npc = interp(y_npc), interp(x_npc)

ax1_1.imshow(y_mt[Sx//2], cmap=color_map_img, vmin=0.0, vmax=vmax_img_mt)
ax1_2.imshow(x_mt[Sx//2], cmap=color_map_img, vmin=0.0, vmax=vmax_img_mt)
ax2_1.imshow(y_mt[:, Sy//2, :], cmap=color_map_img, vmin=0.0, vmax=vmax_img_mt)
ax2_2.imshow(x_mt[:, Sy//2, :], cmap=color_map_img, vmin=0.0, vmax=vmax_img_mt)

ax3_1.imshow(y_npc[Sx//2], cmap=color_map_img, vmin=0.0, vmax=vmax_img_npc)
ax3_2.imshow(x_npc[Sx//2], cmap=color_map_img, vmin=0.0, vmax=vmax_img_npc)
ax4_1.imshow(y_npc[:, Sy//2, :], cmap=color_map_img, vmin=0.0, vmax=vmax_img_npc)
ax4_2.imshow(x_npc[:, Sy//2, :], cmap=color_map_img, vmin=0.0, vmax=vmax_img_npc)

ax1_3.imshow(ker_FP[0][Sx_f//2], cmap=color_map_psf, vmin=0.0, vmax=vmax_psf)       
ax2_3.imshow(ker_FP[0][:, Sy_f//2, :], cmap=color_map_psf, vmin=0.0, vmax=vmax_psf)
ax3_3.imshow(ker_FP[1][Sx_f//2], cmap=color_map_psf, vmin=0.0, vmax=vmax_psf)       
ax4_3.imshow(ker_FP[1][:, Sy_f//2, :], cmap=color_map_psf, vmin=0.0, vmax=vmax_psf) 

plt.savefig(os.path.join(fig_path, 'kernel_fp_real'))
# -----------------------------------------------------------------------------------