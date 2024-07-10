import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ------------------------------------------------------------------------------
dataset_name = 'SimuMix3D_128'
# dataset_name = 'SimuMix3D_256'
std_gauss = [0, 0.5]
poisson   = [0, 1]
scale_factor = 1
ratio = 1
name_net = 'KernelNet'

id_data = 0

path_fig = os.path.join('outputs', 'figures', dataset_name)
ker_FP, ker_BP = [], []

# ------------------------------------------------------------------------------
# load results
for std, poi in zip(std_gauss, poisson):
    path_fig_data = os.path.join('outputs', 'figures', dataset_name,\
        f'scale_{scale_factor}_gauss_{std}_poiss_{poi}_ratio_{ratio}')
    ker_FP.append(io.imread(os.path.join(path_fig_data, 'kernel_fp.tif')))
    ker_BP.append(io.imread(os.path.join(path_fig_data, 'kernel_bp.tif')))

ker_init = io.imread(os.path.join(path_fig_data, 'kernel_init.tif'))
PSF_true = io.imread(os.path.join(path_fig_data, 'kernel_true.tif'))

Nz_kt, Ny_kt, Nx_kt = PSF_true.shape
Nz_kf, Ny_kf, Nx_kf = ker_FP[0].shape
Nz_kb, Ny_kb, Nx_kb = ker_BP[0].shape

y   = io.imread(os.path.join(path_fig,\
    f'scale_{scale_factor}_gauss_{std_gauss[0]}_poiss_{poisson[0]}_ratio_{ratio}',\
    f'sample_{id_data}', 'kernelnet', 'y.tif'))

x_nf = io.imread(os.path.join(path_fig,\
    f'scale_{scale_factor}_gauss_{std_gauss[0]}_poiss_{poisson[0]}_ratio_{ratio}',\
    f'sample_{id_data}', 'kernelnet', 'x.tif'))

x_n = io.imread(os.path.join(path_fig,\
    f'scale_{scale_factor}_gauss_{std_gauss[1]}_poiss_{poisson[1]}_ratio_{ratio}',\
    # f'scale_{scale_factor}_gauss_{std_gauss[1]}_poiss_{poisson}_ratio_0.3',\
    f'sample_{id_data}', 'kernelnet', 'x.tif'))

Nz, Ny, Nx = y.shape

# ------------------------------------------------------------------------------
# FP kernel
# ------------------------------------------------------------------------------
vmax_psf, color_map_psf = 0.01, 'hot'

nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes.ravel(): ax.set_axis_off()

def show_psf(ax, psf):
    ax.imshow(psf, cmap=color_map_psf, vmin=0.0, vmax=vmax_psf)

show_psf(axes[0,0], PSF_true[Nz_kt//2])
show_psf(axes[1,0], PSF_true[:, Ny_kt//2, :])
show_psf(axes[0,1], ker_FP[0][Nz_kf//2])
show_psf(axes[1,1], ker_FP[0][:, Ny_kf//2, :])
show_psf(axes[0,2], ker_FP[1][Nz_kf//2])
show_psf(axes[1,2], ker_FP[1][:, Ny_kf//2, :])

axes[0,0].set_title('True')
axes[0,1].set_title(f'{name_net} (NF)')
axes[0,2].set_title(f'{name_net} (N)')

plt.savefig(os.path.join(path_fig, 'kernel_fp'))

# ------------------------------------------------------------------------------
vmax_psf, color_map_psf = 0.01, 'hot'
vmax_img, color_map_img = y.max()*0.8, 'gray'

nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes.ravel(): ax.set_axis_off()

def img_with_psf_patch(axes, img, psf):
    axes[0].imshow(img[Nz//2], cmap=color_map_img, vmin=0.0, vmax=vmax_img)
    axes[1].imshow(img[:, Ny//2, :], cmap=color_map_img, vmin=0.0,\
        vmax=vmax_img)
    ab_xy = AnnotationBbox(OffsetImage(psf[Nz_kt//2], zoom=2.0,\
        cmap=color_map_psf), xy=(105, 105), pad=0,\
        bboxprops={'edgecolor':'white'})
    axes[0].add_artist(ab_xy)
    ab_xz = AnnotationBbox(OffsetImage(psf[:, Ny_kt//2, :], zoom=2.0,\
        cmap=color_map_psf), xy=(105, 105), pad=0,\
        bboxprops={'edgecolor':'white'})
    axes[1].add_artist(ab_xz)

img_with_psf_patch(axes[:,0], y, PSF_true)
img_with_psf_patch(axes[:,1], x_nf, ker_FP[0])
img_with_psf_patch(axes[:,2], x_n, ker_FP[1])

axes[0,0].set_title('True')
axes[0,1].set_title('Noise Free (NF)')
axes[0,2].set_title('Noisy (N)')

plt.savefig(os.path.join(path_fig, 'kernel_fp_patch'))

# ------------------------------------------------------------------------------
nr, nc = 2, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 2*nr),\
    constrained_layout=True)

colors = ['black', '#6895D2', '#D04848', '#F3B95F']
axes[0].plot(PSF_true[ Nz_kt//2, Ny_kt//2, :], color=colors[0], linewidth=1,\
    label='True')
axes[0].plot(ker_init[ Nz_kt//2, Ny_kt//2, :], color=colors[1], linewidth=1,\
    label='init')
axes[0].plot(ker_FP[0][Nz_kf//2, Ny_kf//2, :], color=colors[2], linewidth=1,\
    label=name_net + '(NF)')
axes[0].plot(ker_FP[1][Nz_kf//2, Ny_kf//2, :], color=colors[3], linewidth=1,\
    label=name_net + '(N)')

axes[1].plot(PSF_true[ :, Ny_kt//2, Nx_kt//2], color=colors[0], linewidth=1,\
    label='True')
axes[1].plot(ker_init[ :, Ny_kt//2, Nx_kt//2], color=colors[1], linewidth=1,\
    label='init')
axes[1].plot(ker_FP[0][:, Ny_kf//2, Nx_kf//2], color=colors[2], linewidth=1,\
    label=name_net + ' (NF)')
axes[1].plot(ker_FP[1][:, Ny_kf//2, Nx_kf//2], color=colors[3], linewidth=1,\
    label=name_net + ' (N)')

axes[0].set_title('PSF profile (x)')
axes[1].set_title('PSF profile (z)')
for ax in axes.ravel(): 
    ax.set_xlabel('Pixel')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.spines[['right', 'top']].set_visible(False)

axes[0].legend(fontsize='x-small', edgecolor='white')

plt.savefig(os.path.join(path_fig, 'kernel_fp_profile'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig, 'kernel_fp_profile.svg'))