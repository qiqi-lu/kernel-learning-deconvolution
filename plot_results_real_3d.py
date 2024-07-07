import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
import numpy as np
import os, torch
from utils import evaluation as eva
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.colors as colors
import utils.dataset_utils as utils_data

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
# data_set_name_test, id_data = 'Microtubule', 0
# data_set_name_test, id_data = 'Nuclear_Pore_complex', 0

data_set_name_test, id_data = 'Microtubule2', 0
# data_set_name_test, id_data = 'Nuclear_Pore_complex2', 0

# ------------------------------------------------------------------------------
name_net = 'kernelnet'
eps = 0.000001

fig_path_data = os.path.join('outputs', 'figures', data_set_name_test.lower(),\
    f'scale_1_gauss_0_poiss_0_ratio_1')
fig_path_sample = os.path.join(fig_path_data, f'sample_{id_data}')

if data_set_name_test in ['Microtubule', 'Nuclear_Pore_complex']:
    fig_path_ker = fig_path_data

if data_set_name_test in ['Microtubule2', 'Nuclear_Pore_complex2']:
    fig_path_ker = os.path.join(fig_path_data, 'kernels_bc_1_re_1')

# ------------------------------------------------------------------------------
# load results
print('load results from :', fig_path_sample)
print('load kernels from :', fig_path_ker)

ker_init = io.imread(os.path.join(fig_path_ker, 'kernel_init.tif'))
ker_true = io.imread(os.path.join(fig_path_ker, 'kernel_true.tif'))
ker_FP   = io.imread(os.path.join(fig_path_ker, 'kernel_fp.tif'))
ker_BP   = io.imread(os.path.join(fig_path_ker, 'kernel_bp.tif'))

if ker_init.shape[-1] == 3:
    ker_init = np.transpose(ker_init, axes=(-1, 0, 1))
    ker_true = np.transpose(ker_true, axes=(-1, 0, 1))
    ker_FP   = np.transpose(ker_FP, axes=(-1, 0, 1))
    ker_BP   = np.transpose(ker_BP, axes=(-1, 0, 1))

y      = io.imread(os.path.join(fig_path_sample, name_net, 'y.tif'))
x      = io.imread(os.path.join(fig_path_sample, name_net, 'x.tif'))
x0     = io.imread(os.path.join(fig_path_sample, name_net, 'x0.tif'))
y_fp   = io.imread(os.path.join(fig_path_sample, name_net, 'y_fp.tif'))
x0_fp  = io.imread(os.path.join(fig_path_sample, name_net, 'x0_fp.tif'))
bp     = io.imread(os.path.join(fig_path_sample, name_net, 'bp.tif'))
y_pred = io.imread(os.path.join(fig_path_sample, name_net, 'y_pred_all.tif'))
y_pred = y_pred[2]

Nz_t, Ny_t, Nx_t = ker_true.shape
Nz_f, Ny_f, Nx_f = ker_FP.shape
Nz_b, Ny_b, Nx_b = ker_BP.shape
Nz, Ny, Nx       = y.shape

# ------------------------------------------------------------------------------
vmax_psf, color_map_psf = 0.01, 'hot'
vmax_img, color_map_img = y.max() * 0.6, 'gray'
vmax_diff = vmax_img

# # ------------------------------------------------------------------------------
# # Show forward kernel image
# # ------------------------------------------------------------------------------
# nr, nc = 2, 3
# fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
#     constrained_layout=True)
# for ax in axes[0:2, 0:2].ravel(): ax.set_axis_off()

# dict_ker = {'cmap': color_map_psf, 'vmin':0.0, 'vmax':vmax_psf}

# axes[0,0].imshow(ker_true[Nz_t//2],       **dict_ker)
# axes[0,1].imshow(ker_FP[Nz_f//2],         **dict_ker)
# axes[1,0].imshow(ker_true[:, Ny_t//2, :], **dict_ker)
# axes[1,1].imshow(ker_FP[:, Ny_f//2, :],   **dict_ker)

# axes[0,2].plot(ker_true[Nz_t//2, :, Nx_t//2], color='black', label='True')
# axes[0,2].plot(ker_init[Nz_t//2, :, Nx_t//2], color='blue', label='init')
# axes[0,2].plot(ker_FP[Nz_f//2, :, Nx_f//2], color='red', label=name_net) 

# axes[1,2].plot(ker_true[:, Ny_t//2, Nx_t//2], color='black', label='True')
# axes[1,2].plot(ker_init[:, Ny_t//2, Nx_t//2], color='blue', label='init')
# axes[1,2].plot(ker_FP[ :, Ny_f//2, Nx_f//2], color='red', label=name_net)

# axes[0,0].set_title('PSF (true) ['+str(np.round(ker_true.sum(), 4)) +']')
# axes[0,1].set_title(f'PSF ({name_net}) ['+str(np.round(ker_FP.sum(), 4)) +']')
# axes[0,2].set_title('PSF profile (xy)')

# axes[1,0].set_title('PSF (true)')
# axes[1,1].set_title(f'PSF ({name_net})')
# axes[1,2].set_title('PSF profile (xz)')

# axes[0,2].set_xlim([0, None]), axes[0,2].set_ylim([0, None])
# axes[1,2].set_xlim([0, None]), axes[1,2].set_ylim([0, None])
# axes[0,2].legend()

# plt.savefig(os.path.join(fig_path_data, 'img_fp'))

# # ------------------------------------------------------------------------------
# # show forward intermediate results
# # ------------------------------------------------------------------------------
# nr, nc = 4, 5
# fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
#     constrained_layout=True)
# [ax.set_axis_off() for ax in axes.ravel()]

# dict_img = {'cmap':color_map_img, 'vmin':0.0, 'vmax':vmax_img}
# dict_img_diff = {'cmap':'gray', 'vmin':0.0, 'vmax':vmax_diff}

# axes[0,0].imshow(y[Nz//2], **dict_img)
# axes[1,0].imshow(x[Nz//2], **dict_img)
# axes[2,0].imshow(y[:, Ny//2, :], **dict_img)
# axes[3,0].imshow(x[:, Ny//2, :], **dict_img)

# axes[0,0].set_title('HR (xy)')
# axes[1,0].set_title('LR (xy)')
# axes[2,0].set_title('HR (xz)')
# axes[3,0].set_title('LR (xz)')

# axes[0,1].imshow(x0[Nz//2], **dict_img)
# axes[1,1].imshow(np.abs(x0-y)[Nz//2], **dict_img_diff)
# axes[2,1].imshow(x0[:, Ny//2, :], **dict_img)
# axes[3,1].imshow(np.abs(x0-y)[:, Ny//2, :], **dict_img_diff)

# axes[0,1].set_title('x0 ({:.2f})'.format(cal_psnr(x0, y)))
# axes[1,1].set_title('|x0-HR|')
# axes[2,1].set_title('x0')
# axes[3,1].set_title('|x0-HR|')

# axes[0,2].imshow(ker_FP[Nz_f//2], **dict_ker)
# axes[1,2].imshow(ker_true[Nz_t//2], **dict_ker)
# axes[2,2].imshow(ker_FP[:, Ny_f//2, :], **dict_ker)
# axes[3,2].imshow(ker_true[:, Ny_t//2, :], **dict_ker)

# axes[0,2].set_title(f'PSF ({name_net}) [' + str(np.round(ker_FP.sum(), 4)) +']')
# axes[1,2].set_title('PSF (true) [' + str(np.round(ker_true.sum(), 4)) +']')
# axes[2,2].set_title(f'PSF ({name_net})')
# axes[3,2].set_title('PSF (true)')

# axes[0,3].imshow(x0_fp[Nz//2], **dict_img)
# axes[2,3].imshow(x0_fp[:, Ny//2, :], **dict_img)

# axes[0,3].set_title('FP(x0)')
# axes[0,3].set_title('FP(x0)')

# axes[0,4].imshow(y_fp[Nz//2], **dict_img)
# axes[1,4].imshow(np.abs(y_fp-x)[Nz//2], **dict_img_diff)
# axes[2,4].imshow(y_fp[:, Ny//2, :], **dict_img)
# axes[3,4].imshow(np.abs(y_fp-x)[:, Ny//2, :], **dict_img_diff)

# axes[0,4].set_title('FP(HR)')
# axes[1,4].set_title('|FP(HR)-LR| ({:.2f})'.format(cal_psnr(y_fp, x)))
# axes[2,4].set_title('FP(HR)')
# axes[3,4].set_title('|FP(HR)-LR|')

# plt.savefig(os.path.join(fig_path_sample, 'img_fp_inter'))

# # ------------------------------------------------------------------------------
# # show FFT of the forward kernel
# # ------------------------------------------------------------------------------
# ker_true_fft = utils_data.fft_n(ker_true, s=y.shape) 
# ker_FP_fft   = utils_data.fft_n(ker_FP, s=y.shape)
# S_fft        = ker_FP_fft.shape
# vmax_fft, color_map_fft = ker_true_fft.max(), 'hot'

# nr, nc = 2, 3
# fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
#     constrained_layout=True)
# for ax in axes[0:2, 0:2].ravel(): ax.set_axis_off()

# dict_ker_fft = {'cmap':color_map_fft, 'vmin':0.0, 'vmax':vmax_fft}

# axes[0,0].set_title('FT(PSF (true))')
# axes[0,1].set_title('FT(PSF (learned))')

# axes[0,0].imshow(ker_true_fft[S_fft[0]//2], **dict_ker_fft)
# axes[0,1].imshow(ker_FP_fft[S_fft[0]//2], **dict_ker_fft)   
# axes[1,0].imshow(ker_true_fft[:, S_fft[1]//2, :], **dict_ker_fft)
# axes[1,1].imshow(ker_FP_fft[:, S_fft[1]//2, :], **dict_ker_fft)

# axes[0,2].plot(ker_true_fft[S_fft[0]//2, :, S_fft[2]//2],\
#     color='black', label='True')
# axes[0,2].plot(ker_FP_fft[S_fft[0]//2, :, S_fft[2]//2],\
#     color='red', label=name_net)
# axes[1,2].plot(ker_true_fft[:, S_fft[1]//2, S_fft[2]//2],\
#     color='black', label='True')
# axes[1,2].plot(ker_FP_fft[ :, S_fft[1]//2, S_fft[2]//2],\
#     color='red', label=name_net)
# axes[0,2].legend()

# plt.savefig(os.path.join(fig_path_data, 'img_fp_fft'))

# # ------------------------------------------------------------------------------
# # show backward intermediate results
# # ------------------------------------------------------------------------------
# nr, nc = 4, 6
# fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
#     constrained_layout=True)
# [ax.set_axis_off() for ax in axes.ravel()]

# axes[0,0].imshow(y[Nz//2], **dict_img)
# axes[1,0].imshow(x[Nz//2], **dict_img)
# axes[2,0].imshow(y[:, Ny//2, :], **dict_img)
# axes[3,0].imshow(x[:, Ny//2, :], **dict_img)
# axes[0,0].set_title('HR (xy)')
# axes[1,0].set_title('LR (xy) ({:.2f}, {:.4f})'\
#     .format(cal_psnr(x, y), cal_ssim(x, y)))
# axes[2,0].set_title('HR (xz)')
# axes[3,0].set_title('LR (xz)')

# axes[0,1].imshow(x0_fp[Nz//2], **dict_img)
# axes[1,1].imshow(np.abs(x0-y)[Nz//2], **dict_img_diff)
# axes[2,1].imshow(x0_fp[:, Ny//2, :], **dict_img)
# axes[3,1].imshow(np.abs(x0-y)[:, Ny//2, :], **dict_img_diff)
# axes[0,1].set_title('FP(x0)')
# axes[1,1].set_title('|x0-HR|')
# axes[2,1].set_title('FP(x0)')
# axes[3,1].set_title('|x0-HR|')

# dict_dv = {'cmap':'seismic', 'vmin':0.5, 'vmax':1.5}

# axes[0,2].imshow((x/(x0_fp + eps))[Nz//2], **dict_dv)
# axes[1,2].imshow(np.abs(y_fp-x)[Nz//2], **dict_img_diff)
# axes[2,2].imshow((x/(x0_fp + eps))[:, Ny//2, :], **dict_dv)
# axes[3,2].imshow(np.abs(y_fp-x)[:, Ny//2, :], **dict_img_diff)

# axes[0,2].set_title('LR/FP(x0)')
# axes[1,2].set_title('|FP(HR)-LR|')
# axes[2,2].set_title('LR/FP(x0)')
# axes[3,2].set_title('|FP(HR)-LR|')

# axes[0,3].imshow(ker_BP[Nz_b//2], cmap=color_map_psf, vmin=0.0, vmax=np.max(ker_BP))
# axes[2,3].imshow(ker_BP[:, Ny_b//2, :], cmap=color_map_psf, vmin=0.0, vmax=np.max(ker_BP))

# axes[0,3].set_title('BP kernel [{:.2f}]'.format(ker_BP.sum()))
# axes[2,3].set_title('BP kernel')

# axes[0,4].imshow(bp[Nz//2], cmap='seismic', vmin=0.0, vmax=2.0)
# axes[2,4].imshow(bp[:, Ny//2, :], cmap='seismic', vmin=0.0, vmax=2.0)

# axes[0,4].set_title('BP(LR/FP(x0))')
# axes[2,4].set_title('BP(LR/FP(x0))')

# axes[0,5].imshow(y_pred[Nz//2], **dict_img)
# axes[1,5].imshow(np.abs(y_pred-y)[Nz//2], **dict_img_diff)   
# axes[2,5].imshow(y_pred[:, Ny//2, :], **dict_img)
# axes[3,5].imshow(np.abs(y_pred-y)[:, Ny//2, :], **dict_img_diff)

# axes[0,5].set_title('xk ({:.4f})'.format(cal_ssim(y_pred, y)))
# axes[1,5].set_title('|xk-HR| ({:.2f})'.format(cal_psnr(y_pred, y)))
# axes[2,5].set_title('xk')
# axes[3,5].set_title('|xk-HR|') 

# plt.savefig(os.path.join(fig_path_sample, 'img_bp_inter'))

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
print('-'*80)
print('plot restored images ...')
data = []
data.append(x)
data.append(io.imread(os.path.join(fig_path_sample, 'deconvblind',\
    'deconv.tif')))
data.append(io.imread(os.path.join(fig_path_sample, 'traditional',\
    'deconv_20.tif')))
data.append(y_pred)
data.append(y)
data = np.array(data)

N_meth = data.shape[0] # number of methods (include raw and gt)

# ------------------------------------------------------------------------------
print('-'*80)
print('PSNR | SSIM:')
for i in range(N_meth-1):
    print(cal_psnr(data[i], data[-1]), cal_ssim(data[i], data[-1]))
print('-'*80)

# ------------------------------------------------------------------------------
if data_set_name_test in ['Nuclear_Pore_complex', 'Nuclear_Pore_complex2']: 
    pos, size = [200,300], [150,300] # 0
    cmap = colors.LinearSegmentedColormap.from_list("", \
        ["black","#21D416","white"]) # green
    color_map_img, color_box_edge = cmap, 'white'

if data_set_name_test in ['Microtubule', 'Microtubule2']:
    pos, size = [200, 200], [150,300] # 0
    cmap = colors.LinearSegmentedColormap.from_list("", \
        ["black","#03AED2","white"]) # blue
    color_map_img, color_box_edge = cmap, 'white'

dict_img = {'cmap':color_map_img, 'vmin':0.0, 'vmax':data[-1].max()*0.6}

# ------------------------------------------------------------------------------
def interp(x):
    ps_xy, ps_z = 25, 160
    z_scale = ps_z/ps_xy
    x = torch.tensor(x)[None, None]
    x = torch.nn.functional.interpolate(x, scale_factor=(z_scale, 1, 1),\
        mode='nearest')
    x = x.numpy()[0,0]
    return x

# ------------------------------------------------------------------------------
nr, nc = 3, 5
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

for ax in axes.ravel():
    ax.set_axis_off()

id_slice = 1
for i in range(N_meth):
    # whole slice
    img = data[i][id_slice]
    axes[0,i].imshow(img, **dict_img)
    box = patches.Rectangle(xy=(pos[1], pos[0]), width=size[1], height=size[0],\
        fill=False, edgecolor=color_box_edge)
    axes[0,i].add_patch(box)
    # pacth
    patch = img[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]]
    axes[1,i].imshow(patch, **dict_img)
    # xz plane of pacth
    img_inter = interp(data[i])
    patch_inter = img_inter[:, pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]]
    Nz_patch, Ny_patch, Nx_patch = patch_inter.shape
    axes[2,i].imshow(patch_inter[:, Ny_patch//2, :], **dict_img)

plt.savefig(os.path.join(fig_path_sample, 'image_restored.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(fig_path_sample, 'image_restored.svg'))