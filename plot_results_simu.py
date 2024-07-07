import matplotlib.pyplot as plt
import utils.evaluation as eva
import utils.dataset_utils as utils_data
import skimage.io as io
import skimage.exposure as exposure
import numpy as np
import os
from utils import evaluation as eva
from skimage.measure import profile_line

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

def cal_ncc(x, y):
    return eva.NCC(img_true=y, img_test=x)

# ------------------------------------------------------------------------------
data_set_name_test = 'SimuMix3D_128'
# data_set_name_test = 'SimuMix3D_256'
# data_set_name_test = 'SimuMix3D_382'
# ------------------------------------------------------------------------------
# std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 1
std_gauss, poisson, ratio = 0, 0, 1

id_sample = 0

num_data  = 1
id_repeat = 1

# methods_iter = [100, 100, 100, 30]
# methods_iter = [30, 30, 30, 30]
methods_iter = [2, 2, 2, 2]

# ------------------------------------------------------------------------------
scale_factor = 1
name_net = 'kernelnet'
num_iter_train = 2
eps = 0.000001
# ------------------------------------------------------------------------------
path_fig_data = os.path.join('outputs', 'figures', data_set_name_test,\
    f'scale_{scale_factor}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}')
path_fig_ker = os.path.join(path_fig_data,\
    f'kernels_bc_{num_data}_re_{id_repeat}')
path_fig_sample = os.path.join(path_fig_data, f'sample_{id_sample}')

print('-'*80)
print('load results from :', path_fig_sample)
print('load kernels from :', path_fig_ker)

# ------------------------------------------------------------------------------
# load kernels and results of KLD
# ------------------------------------------------------------------------------
print('-'*80)
print('load kernels ...')
ker_init = io.imread(os.path.join(path_fig_ker, 'kernel_init.tif'))
ker_true = io.imread(os.path.join(path_fig_ker, 'kernel_true.tif'))
ker_FP   = io.imread(os.path.join(path_fig_ker, 'kernel_fp.tif'))
ker_BP   = io.imread(os.path.join(path_fig_ker, 'kernel_bp.tif'))

print('load results of KLD ...')
y     = io.imread(os.path.join(path_fig_sample, name_net, 'y.tif'))
x     = io.imread(os.path.join(path_fig_sample, name_net, 'x.tif'))
x0    = io.imread(os.path.join(path_fig_sample, name_net, 'x0.tif'))
y_fp  = io.imread(os.path.join(path_fig_sample, name_net, 'y_fp.tif'))
x0_fp = io.imread(os.path.join(path_fig_sample, name_net, 'x0_fp.tif'))
bp    = io.imread(os.path.join(path_fig_sample, name_net, 'bp.tif'))

y_pred_all = io.imread(os.path.join(path_fig_sample, name_net,\
    'y_pred_all.tif'))
num_iter_train = 2
y_pred = y_pred_all[num_iter_train]

# ------------------------------------------------------------------------------
Nz_kt, Ny_kt, Nx_kt = ker_true.shape
Nz_kf, Ny_kf, Nx_kf = ker_FP.shape
Nz_kb, Ny_kb, Nx_kb = ker_BP.shape
Nz, Ny, Nx = y.shape

# ------------------------------------------------------------------------------
vmax_psf, color_map_psf = ker_true.max(), 'hot'
vmax_img, color_map_img = y.max(), 'gray'
vmax_diff = vmax_img

dict_ker = {'cmap':color_map_psf, 'vmin':0.0, 'vmax':vmax_psf}
dict_ker_profile = {'linewidth': 0.5}
dict_img = {'cmap': color_map_img, 'vmin': 0.0, 'vmax': vmax_img}
dict_img_diff = {'cmap': 'gray', 'vmin': 0.0, 'vmax': vmax_diff}

# ------------------------------------------------------------------------------
# Show forward kernel image
# ------------------------------------------------------------------------------
print('plot forward kernel ...')
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes[0:2, 0:2].ravel(): ax.set_axis_off()

# ker = exposure.adjust_log(ker, gain=2.0)
    
axes[0,0].imshow(ker_true[Nz_kt//2], **dict_ker)
axes[0,1].imshow(ker_FP[Nz_kf//2], **dict_ker)
axes[1,0].imshow(ker_true[:, Ny_kt//2, :], **dict_ker)
axes[1,1].imshow(ker_FP[:, Ny_kf//2, :], **dict_ker)

axes[0,2].plot(ker_true[Nz_kt//2, :, Nx_kt//2], color='black', label='True',\
    **dict_ker_profile)
axes[0,2].plot(ker_init[Nz_kt//2, :, Nx_kt//2], color='blue',  label='init',\
    **dict_ker_profile)
axes[0,2].plot(ker_FP  [Nz_kf//2, :, Nx_kf//2], color='red',   label=name_net,\
    **dict_ker_profile) 

axes[1,2].plot(ker_true[:, Ny_kt//2, Nx_kt//2], color='black', label='True',\
    **dict_ker_profile)
axes[1,2].plot(ker_init[:, Ny_kt//2, Nx_kt//2], color='blue',  label='init',\
    **dict_ker_profile)
axes[1,2].plot(ker_FP  [:, Ny_kf//2, Nx_kf//2], color='red',   label=name_net,\
    **dict_ker_profile)

axes[0,0].set_title(f'PSF (true) [{str(np.round(ker_true.sum(), 4))}]')
axes[0,1].set_title(f'PSF ({name_net}) [' + str(np.round(ker_FP.sum(), 4)) +']')
axes[0,2].set_title('PSF profile (xy)')

axes[1,0].set_title('PSF (true)')
axes[1,1].set_title(f'PSF ({name_net})')
axes[1,2].set_title('PSF profile (xz)')

axes[0,2].set_xlim([0, None]), axes[0,2].set_ylim([0, None])
axes[1,2].set_xlim([0, None]), axes[1,2].set_ylim([0, None])
axes[0,2].legend()

plt.savefig(os.path.join(path_fig_ker, 'img_fp'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_ker, 'img_fp.svg'))

# ------------------------------------------------------------------------------
# show forward intermediate results
# ------------------------------------------------------------------------------
print('plot forward intermediate results ...')
nr, nc = 4, 5
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4*nc, 2.4*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,0].imshow(y[Nz//2], **dict_img)
axes[1,0].imshow(x[Nz//2], **dict_img)
axes[2,0].imshow(y[:, Ny//2, :], **dict_img)
axes[3,0].imshow(x[:, Ny//2, :], **dict_img)

axes[0,0].set_title('HR (xy)')
axes[1,0].set_title('LR (xy)')
axes[2,0].set_title('HR (xz)')
axes[3,0].set_title('LR (xz)')

diff_x0_y = np.abs(x0-y)
axes[0,1].imshow(x0[Nz//2],              **dict_img)
axes[1,1].imshow(diff_x0_y[Nz//2],       **dict_img_diff)
axes[2,1].imshow(x0[:, Ny//2, :],        **dict_img)
axes[3,1].imshow(diff_x0_y[:, Ny//2, :], **dict_img_diff)

axes[0,1].set_title('x0 ({:.2f})'.format(cal_psnr(x0, y)))
axes[1,1].set_title('|x0-HR|')
axes[2,1].set_title('x0')
axes[3,1].set_title('|x0-HR|')

axes[0,2].imshow(ker_FP[Nz_kf//2], **dict_ker)
axes[1,2].imshow(ker_true[Nz_kt//2], **dict_ker)
axes[2,2].imshow(ker_FP[:, Ny_kf//2, :], **dict_ker)
axes[3,2].imshow(ker_true[:, Ny_kt//2, :], **dict_ker)

axes[0,2].set_title(f'PSF ({name_net}) [' + str(np.round(ker_FP.sum(), 4)) +']')
axes[1,2].set_title('PSF (true) [' + str(np.round(ker_true.sum(), 4)) +']')
axes[2,2].set_title(f'PSF ({name_net})')
axes[3,2].set_title('PSF (true)')

axes[0,3].imshow(x0_fp[Nz//2], **dict_img)
axes[2,3].imshow(x0_fp[:,Ny//2,:], **dict_img)
axes[0,3].set_title('FP(x0)')
axes[0,3].set_title('FP(x0)')

diff_yfp_x = np.abs(y_fp-x)

axes[0,4].imshow(y_fp[Nz//2], **dict_img)
axes[1,4].imshow(diff_yfp_x[Nz//2], **dict_img_diff)
axes[2,4].imshow(y_fp[:, Ny//2, :], **dict_img)
axes[3,4].imshow(diff_yfp_x[:, Ny//2, :], **dict_img_diff)

axes[0,4].set_title('FP(HR)')
axes[1,4].set_title('|FP(HR)-LR| ({:.2f})'.format(cal_psnr(y_fp, x)))
axes[2,4].set_title('FP(HR)')
axes[3,4].set_title('|FP(HR)-LR|')

plt.savefig(os.path.join(path_fig_sample, 'img_fp_inter'))

# ------------------------------------------------------------------------------
# show FFT of the forward kernel
# ------------------------------------------------------------------------------
print('plot fft of forward kernel ...')
s_fft = y.shape
ker_true_fft = utils_data.fft_n(ker_true, s=s_fft) 
ker_FP_fft   = utils_data.fft_n(ker_FP, s=s_fft)

Nz_kf_ft, Ny_kf_ft, Nx_kf_ft = ker_FP_fft.shape

dict_kf_fft = {'cmap': 'hot', 'vmin': 0.0, 'vmax': ker_true_fft.max()}
dict_kf_fft_profile = {'linewidth': 0.5}
# ------------------------------------------------------------------------------
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes[0:2, 0:2].ravel(): ax.set_axis_off()

axes[0,0].set_title('FT(PSF (true))')
axes[0,1].set_title('FT(PSF (learned))')

axes[0,0].imshow(ker_true_fft[Nz_kf_ft//2], **dict_kf_fft)
axes[0,1].imshow(ker_FP_fft[Nz_kf_ft//2], **dict_kf_fft)   
axes[1,0].imshow(ker_true_fft[:, Ny_kf_ft//2, :], **dict_kf_fft)
axes[1,1].imshow(ker_FP_fft[:, Ny_kf_ft//2, :], **dict_kf_fft)

axes[0,2].plot(ker_true_fft[Nz_kf_ft//2, :, Nx_kf_ft//2], color='black',\
    label='True', **dict_kf_fft_profile)
axes[0,2].plot(ker_FP_fft[Nz_kf_ft//2, :, Nx_kf_ft//2], color='red',\
    label=name_net, **dict_kf_fft_profile)
axes[1,2].plot(ker_true_fft[:, Ny_kf_ft//2, Nx_kf_ft//2], color='black',\
    label='True', **dict_kf_fft_profile)
axes[1,2].plot(ker_FP_fft[ :, Ny_kf_ft//2, Nx_kf_ft//2], color='red',\
    label=name_net, **dict_kf_fft_profile)
axes[0,2].legend()

plt.savefig(os.path.join(path_fig_ker, 'img_fp_fft'))

# ------------------------------------------------------------------------------
# show backward intermediate results
# ------------------------------------------------------------------------------
print('plot backward intermediate results ...')
nr, nc = 4, 6
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,0].imshow(y[Nz//2], **dict_img)
axes[1,0].imshow(x[Nz//2], **dict_img)
axes[2,0].imshow(y[:, Ny//2, :], **dict_img)
axes[3,0].imshow(x[:, Ny//2, :], **dict_img)

axes[0,0].set_title('HR (xy)')
axes[1,0].set_title('LR (xy) ({:.2f}, {:.4f})'\
    .format(cal_psnr(x, y), cal_ssim(x, y)))
axes[2,0].set_title('HR (xz)')
axes[3,0].set_title('LR (xz)')

diff_x0_y = np.abs(x0-y)

axes[0,1].imshow(x0_fp[Nz//2], **dict_img)
axes[1,1].imshow(diff_x0_y[Nz//2], **dict_img_diff)
axes[2,1].imshow(x0_fp[:,Ny//2,:], **dict_img)
axes[3,1].imshow(diff_x0_y[:, Ny//2, :], **dict_img_diff)

axes[0,1].set_title('FP(x0)')
axes[1,1].set_title('|x0-HR|')
axes[2,1].set_title('FP(x0)')
axes[3,1].set_title('|x0-HR|')

diff_yfp_x  = np.abs(y_fp-x)
div_x_x0_fp = x/(x0_fp + eps)

dict_dv = {'cmap': 'seismic', 'vmin': 0.5, 'vmax': 1.5}

dvax = axes[0,2].imshow(div_x_x0_fp[Nz//2], **dict_dv)
axes[1,2].imshow(diff_yfp_x[Nz//2], **dict_img_diff)
axes[2,2].imshow(div_x_x0_fp[:, Ny//2, :], **dict_dv)
axes[3,2].imshow(diff_yfp_x[:, Ny//2, :], **dict_img_diff)

axes[0,2].set_title('LR/FP(x0)')
axes[1,2].set_title('|FP(HR)-LR|')
axes[2,2].set_title('LR/FP(x0)')
axes[3,2].set_title('|FP(HR)-LR|')

fig.colorbar(dvax, ax=axes[0,2], location='bottom')

dict_kb = {'cmap': color_map_psf, 'vmin': 0.0, 'vmax': np.max(ker_BP)}

axes[0,3].imshow(ker_BP[Nz_kb//2], **dict_kb)
axes[2,3].imshow(ker_BP[:, Ny_kb//2, :], **dict_kb)
axes[0,3].set_title('BP kernel [{:.2f}]'.format(ker_BP.sum()))
axes[2,3].set_title('BP kernel')

dict_bp = {'cmap': 'seismic', 'vmin': 0.0, 'vmax': 2.0}

axes[0,4].imshow(bp[Nz//2], **dict_bp)
axes[2,4].imshow(bp[:, Ny//2, :], **dict_bp)
axes[0,4].set_title('BP(LR/FP(x0))')
axes[2,4].set_title('BP(LR/FP(x0))')

diff_ypred_y = np.abs(y_pred-y)

axes[0,5].imshow(y_pred[Nz//2], **dict_img)      
axes[1,5].imshow(diff_ypred_y[Nz//2], **dict_img_diff)     
axes[2,5].imshow(y_pred[:,Ny//2,:], **dict_img) 
axes[3,5].imshow(diff_ypred_y[:,Ny//2,:], **dict_img_diff)

axes[0,5].set_title('xk ({:.4f})'.format(cal_ssim(y_pred, y)))
axes[1,5].set_title('|xk-HR| ({:.2f})'.format(cal_psnr(y_pred, y)))
axes[2,5].set_title('xk')
axes[3,5].set_title('|xk-HR|')

plt.savefig(os.path.join(path_fig_sample, 'img_bp_inter'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'img_bp_inter.svg'))

# os._exit(0)
# ------------------------------------------------------------------------------
# load reaults of conventional methods
# ------------------------------------------------------------------------------
print('load results of conventional methods ...')
methods_name  = ['traditional', 'gaussian', 'butterworth', 'wiener_butterworth']
methods_color = ['#D04848', '#007F73', '#4CCD99', '#FFC700', '#FFF455']

def load_result(path, name, iter):
    out = []
    out.append(io.imread(os.path.join(path, name, f'deconv_{iter}.tif')))
    out.append(io.imread(os.path.join(path, name, 'deconv_bp.tif')))
    out.append(np.load(os.path.join(path, name, f'deconv_metrics_{iter}.npy')))
    return out

# load results from conventional methods
out_trad = load_result(path_fig_sample, methods_name[0], methods_iter[0])
out_gaus = load_result(path_fig_sample, methods_name[1], methods_iter[1])
out_butt = load_result(path_fig_sample, methods_name[2], methods_iter[2])
out_wien = load_result(path_fig_sample, methods_name[3], methods_iter[3])

# ------------------------------------------------------------------------------
# show backward kernel image
# ------------------------------------------------------------------------------
print('plot backward kernels ...')
nr, nc = 3, 10
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

def show(psf, ker_bp, axes, s=None, title=''):
    psf_fft    = utils_data.fft_n(psf, s=s)
    bp_ker_fft = utils_data.fft_n(ker_bp, s=s)
    Nz_k, Ny_k, Nx_k = psf.shape
    a = np.abs(bp_ker_fft)
    b = np.abs(psf_fft*bp_ker_fft)

    axes[0,0].set_title(f'BP ({title}) (xy)')
    axes[0,1].set_title(f'BP ({title}) (xz)')
    axes[1,0].set_title('FT(BP) (xy)')
    axes[1,1].set_title('FT(BP) (xz)')
    axes[2,0].set_title('|FT(FP) x FT(BP)| (xy)')
    axes[2,1].set_title('|FT(FP) x FT(BP)| (xz)')

    dict_tmp = {'cmap': 'hot', 'vmin': 0.0}

    axes[0,0].imshow(ker_bp[Nz_k//2],       vmax=ker_bp.max(), **dict_tmp)    
    axes[0,1].imshow(ker_bp[:, Ny_k//2, :], vmax=ker_bp.max(), **dict_tmp)
    axes[1,0].imshow(a[Nz_kf_ft//2],        vmax=a.max(), **dict_tmp)  
    axes[1,1].imshow(a[:, Ny_kf_ft//2, :],  vmax=a.max(), **dict_tmp)
    axes[2,0].imshow(b[Nz_kf_ft//2],        vmax=b.max(), **dict_tmp)   
    axes[2,1].imshow(b[:, Ny_kf_ft//2, :],  vmax=b.max(), **dict_tmp)

show(ker_true, out_trad[1], axes=axes[:,0:2], s=s_fft, title='Traditional')
show(ker_true, out_gaus[1], axes=axes[:,2:4], s=s_fft, title='Gaussian')
show(ker_true, out_butt[1], axes=axes[:,4:6], s=s_fft, title='Butterworth')
show(ker_true, out_wien[1], axes=axes[:,6:8], s=s_fft, title='WB')
show(ker_FP, ker_BP, axes=axes[:,8: ], s=s_fft, title='KLD')

print('-'*80)
print('sum of kf: ', ker_FP.sum())
print('sum of kb: ', ker_BP.sum())
print('-'*80)

plt.savefig(os.path.join(path_fig_ker, 'img_bp.png'))

# ------------------------------------------------------------------------------
# plot FFT of backward kernels
# ------------------------------------------------------------------------------
print('plot fft of backward kernels ...')
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

def plot_profile(axes, psf, ker_bp, s=None, color=None, label=None):
    psf_fft = utils_data.fft_n(psf, s=s)
    bp_ker_fft = utils_data.fft_n(ker_bp, s=s)
    Nz_k, Ny_k, Nx_k = ker_bp.shape
    a = np.abs(bp_ker_fft)
    b = np.abs(psf_fft*bp_ker_fft)

    axes[0,0].plot(ker_bp[Nz_k//2, Ny_k//2, :], color=color, label=label)
    axes[0,1].plot(a[Nz_kf_ft//2, Ny_kf_ft//2, Nx_kf_ft//2:], color=color,\
        label=label)
    axes[0,2].plot(b[Nz_kf_ft//2, Ny_kf_ft//2, Nx_kf_ft//2:], color=color,\
        label=label)

    axes[1,0].plot(ker_bp[:, Ny_k//2, Nx_k//2], color=color, label=label)
    axes[1,1].plot(a[Nz_kf_ft//2:, Ny_kf_ft//2, Nx_kf_ft//2], color=color,\
        label=label)
    axes[1,2].plot(b[Nz_kf_ft//2:, Ny_kf_ft//2, Nx_kf_ft//2], color=color,\
        label=label)

axes[0,0].axhline(y=0.0, color='black')
axes[1,0].axhline(y=0.0, color='black')
axes[0,0].set_title('BP (x)')
axes[0,1].set_title('|FT(BP)| (x)')
axes[0,2].set_title('|FT(FP) x FT(BP)| (x)')
axes[1,0].set_title('BP (z)')
axes[1,1].set_title('|FT(BP)| (z)')
axes[1,2].set_title('|FT(FP) x FT(BP)| (z)')

plot_profile(axes, psf=ker_true, ker_bp=out_trad[1], s=s_fft,\
    color=methods_color[4], label=methods_name[0])
plot_profile(axes, psf=ker_true, ker_bp=out_gaus[1], s=s_fft,\
    color=methods_color[3], label=methods_name[1])
plot_profile(axes, psf=ker_true, ker_bp=out_butt[1], s=s_fft,\
    color=methods_color[2], label=methods_name[2])
plot_profile(axes, psf=ker_true, ker_bp=out_wien[1], s=s_fft,\
    color=methods_color[1], label=methods_name[3])
plot_profile(axes, psf=ker_FP, ker_bp=ker_BP, s=s_fft,\
    color=methods_color[0], label=name_net)

axes[1,2].legend()

for ax in axes[:,1:].ravel():
    ax.set_xlim([0, None]), ax.set_ylim([0, None])
    ax.set_xlabel('Frequency'), ax.set_ylabel('Normalized value')
    
plt.savefig(os.path.join(path_fig_ker, 'profile_bp_fft'))

# ------------------------------------------------------------------------------
# show the restored images
# ------------------------------------------------------------------------------
print('-'*80)
print('plot restored images ...')
pos_text_x, pos_text_y = 5, 10
line_start_xy, line_end_xy = (38, 34), (56, 19)
line_start_xz, line_end_xz = (86, 44), (86, 64)
id_slice = Nz//2
# ------------------------------------------------------------------------------
nr, nc = 4, 7
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

dict_text  = {'color': 'white', 'fontsize': 'x-large'}
dict_line  = {'color': 'white', 'linewidth': 1}
dict_image = {'cmap': 'gray', 'vmin': 0, 'vmax': vmax_img}
dict_image_diff = {'cmap': 'gray', 'vmin': 0, 'vmax': vmax_diff}

axes[0,0].text(pos_text_x, pos_text_y, 'RAW (xy) ({:>.2f}, {:>.4f})'\
    .format(cal_psnr(x, y), cal_ssim(x, y)),\
    **dict_text)
axes[0,6].text(pos_text_x, pos_text_y, 'GT (xy)',   **dict_text)
axes[2,0].text(pos_text_x, pos_text_y, 'RAW (xz)',  **dict_text)
axes[2,6].text(pos_text_x, pos_text_y, 'GT (xz)',   **dict_text)

# RAW
axes[0,0].imshow(x[id_slice], **dict_image)
axes[2,0].imshow(x[:, Ny//2, :], **dict_image)
axes[0,0].plot((line_start_xy[0], line_end_xy[0]),\
    (line_start_xy[1], line_end_xy[1]), **dict_line)
axes[2,0].plot((line_start_xz[0], line_end_xz[0]),\
    (line_start_xz[1], line_end_xz[1]), **dict_line)
print('RAW', 'PSNR:', cal_psnr(x, y))

axes[0,6].imshow(y[id_slice], **dict_image)
axes[2,6].imshow(y[:, Ny//2, :], **dict_image)

# Traditional, Gaussian, Butterworth, WB
def show_result(out, axes, name):
    diff = np.abs(out[0]-y)
    num_iter = out[-1].shape[0] - 1
    axes[0].text(pos_text_x, pos_text_y, '{} {:d} it'.format(name, num_iter),\
        **dict_text)
    axes[1].text(pos_text_x, pos_text_y, '({:>.2f}, {:>.4f})'\
        .format(out[-1][-1,0], out[-1][-1,1]), **dict_text)
    axes[0].imshow(out[0][id_slice],    **dict_image)
    axes[1].imshow(diff[id_slice],      **dict_image_diff)
    axes[2].imshow(out[0][:, Ny//2, :], **dict_image)
    axes[3].imshow(diff[:, Ny//2, :],   **dict_image_diff)
    print(name, 'PSNR:', cal_psnr(out[0], y))

show_result(out_trad, axes[0:4,1], name='Traditional')
show_result(out_gaus, axes[0:4,2], name='Gaussian')
show_result(out_butt, axes[0:4,3], name='Butterworth')
show_result(out_wien, axes[0:4,4], name='WB')

# KLD
axes[0,5].text(pos_text_x, pos_text_y, '{} {:d} it'\
    .format('KLD', num_iter_train), **dict_text)
axes[0,5].imshow(y_pred[id_slice], **dict_image)
axes[1,5].text(pos_text_x, pos_text_y, '({:>.2f}, {:>.4f})'\
    .format(cal_psnr(y_pred, y), cal_ssim(y_pred, y)), **dict_text)
axes[1,5].imshow(np.abs(y_pred-y)[id_slice], **dict_image_diff)
diff_ypred_y = np.abs(y_pred-y)
axes[2,5].imshow(y_pred[:, Ny//2, :], **dict_image)
axes[3,5].imshow(diff_ypred_y[:, Ny//2, :], **dict_image_diff)
print('KLD', 'PSNR:', cal_psnr(y_pred, y))

io.imsave(fname=os.path.join(path_fig_sample, 'xz.tif'),\
    arr=y_pred[:, Ny//2, :], check_contrast=False)
io.imsave(fname=os.path.join(path_fig_sample, 'xy.tif'),\
    arr=y_pred[id_slice], check_contrast=False)

plt.savefig(os.path.join(path_fig_sample, 'img_restored.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'img_restored.svg'))
print('-'*80)

# ------------------------------------------------------------------------------
# profile line
# ------------------------------------------------------------------------------
print('-'*80)
print('plot profiel lines ...')
methods_name  = ['Traditional', 'Gaussian', 'Butterworth', 'WB']
colors = ['#B3BE9D', '#FDE767', '#F3B95F', '#E28154']
# ['KernelNet', 'WB', 'Butterworth', 'Gaussian', 'Traditional', 'RAW']
# ['#D04848', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D','#6895D2']

# ------------------------------------------------------------------------------
nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

dict_profile = {'linewidth': 0.5}

profiles_xy, profiles_xz = [], []
for i, ax in enumerate(axes.ravel()):
    # line in xy plane
    if i == 0:
        line_start = (line_start_xy[1], line_start_xy[0])
        line_end   = (line_end_xy[1],   line_end_xy[0])
        # RAW
        profile = profile_line(x[id_slice], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='RAW', color='#2A629A', **dict_profile)
        profiles_xy.append(profile.tolist())
        # Traditional, Gaussian, Butterworth, WB
        for out, name, color in zip([out_trad, out_gaus, out_butt, out_wien],\
            methods_name, colors):
            profile = profile_line(out[0][id_slice], line_start, line_end,\
                linewidth=1)
            ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xy.append(profile.tolist())
        # KLD
        profile = profile_line(y_pred[id_slice], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='KLD', color='red', **dict_profile)
        profiles_xy.append(profile.tolist())
        # GT
        profile = profile_line(y[id_slice], line_start, line_end, linewidth=1)
        ax.plot(profile, label='GT', color='black', linestyle='--',\
            **dict_profile)
        profiles_xy.append(profile.tolist())

    # line in xz plane
    if i == 1:
        line_start = (line_start_xz[1], line_start_xz[0])
        line_end   = (line_end_xz[1],   line_end_xz[0])
        # RAW
        profile = profile_line(x[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='RAW', color='#2A629A', **dict_profile)
        profiles_xz.append(profile.tolist())
        # Traditional, Gaussian, Butterworth, WB
        for out, name, color in zip([out_trad, out_gaus, out_butt, out_wien],\
            methods_name, colors):
            profile = profile_line(out[0][:, Ny//2, :], line_start, line_end,\
                linewidth=1)
            ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xz.append(profile.tolist())
        # KLD
        profile = profile_line(y_pred[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='KLD', color='red', **dict_profile)
        profiles_xz.append(profile.tolist())
        # GT
        profile = profile_line(y[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='GT', color='black', linestyle='--',\
            **dict_profile)
        profiles_xz.append(profile.tolist())

    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_linewidth(0.5)
        ax.tick_params(width=0.5)
    ax.tick_params(direction='in')
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))
    ax.set_ylabel('Intensity (AU)')
    ax.set_xlabel('Distance (pixel)')

print('-'*80)
print('Vallue of line profiels:')
print('-'*80)
print(tabulate(profiles_xy))
print('-'*80)
print(tabulate(profiles_xz))
print('-'*80)

plt.legend(fontsize='xx-small')
plt.savefig(os.path.join(path_fig_sample, 'img_restored_profile.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'img_restored_profile.svg'))

# ------------------------------------------------------------------------------
# show metrics curve
# ------------------------------------------------------------------------------
print('plot metrics ...')
psnrs, ssims, nccs = [], [], []

for i in range(len(y_pred_all)):
    psnrs.append(cal_psnr(y_pred_all[i], y))
    ssims.append(cal_ssim(y_pred_all[i], y))
    nccs.append(cal_ncc(y_pred_all[i], y))
mtrics_kld = np.stack([psnrs, ssims, nccs]).transpose()
# ------------------------------------------------------------------------------
nr, nc = 1, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

dict_line_metrics = {'linestyle': '-', 'marker':'.', 'markersize':2,\
    'linewidth':0.5}
dict_line_axh     = {'linestyle': '--', 'linewidth':0.5}

# methods_color = ['red', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D',\
#     '#6895D2']

for i in range(3): # (PSNR, SSIM, NCC)
    axes[i].plot(out_trad[-1][:,i], color=methods_color[4], label='Traditional',\
        **dict_line_metrics)
    axes[i].plot(out_gaus[-1][:,i], color=methods_color[3], label='Gaussian',\
        **dict_line_metrics)
    axes[i].plot(out_butt[-1][:,i], color=methods_color[2], label='Butterworth',\
        **dict_line_metrics)
    axes[i].plot(out_wien[-1][:,i], color=methods_color[1], label='WB',\
        **dict_line_metrics)
    axes[i].plot(mtrics_kld[:,i],   color=methods_color[0], label='KLD',\
        **dict_line_metrics)

    axes[i].axhline(y=mtrics_kld[num_iter_train,i], color=methods_color[0],\
        **dict_line_axh)
    axes[i].axhline(y=out_wien[-1][num_iter_train,i], color=methods_color[1],\
        **dict_line_axh)

    print('-'*80)
    print(tabulate([out_trad[-1][:,i], out_gaus[-1][:,i], out_butt[-1][:,i],\
        out_wien[-1][:,i], mtrics_kld[:,i]]))
    print('-'*80)

    axes[i].spines[['right', 'top']].set_visible(False)
    axes[i].set_xlabel('Iteration Number')
    axes[i].set_xlim([0, None])
    axes[i].legend(edgecolor='white')

axes[0].set_ylabel('PSNR')
axes[1].set_ylabel('SSIM')
axes[2].set_ylabel('NCC')

plt.savefig(os.path.join(path_fig_sample, 'curve_metrics.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'curve_metrics.svg'))
# ------------------------------------------------------------------------------