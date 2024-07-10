import matplotlib.pyplot as plt
import utils.evaluation as eva
import utils.dataset_utils as utils_data
import skimage.io as io
import skimage.exposure as exposure
import numpy as np
import os
from utils import evaluation as eva
from skimage.measure import profile_line

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
std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0, 0, 1

id_sample = 1

num_data  = 1
id_repeat = 1

# methods_iter   = [30, 30, 30, 2]
methods_iter   = [2, 2, 2, 2]
num_iter_train = 2

# ------------------------------------------------------------------------------
scale_factor = 1
name_net = 'kernelnet_ss'
num_iter_train = 2
eps = 0.000001

# ------------------------------------------------------------------------------
path_fig_data = os.path.join('outputs', 'figures', data_set_name_test,\
    f'scale_{scale_factor}_gauss_{std_gauss}_poiss_{poisson}_ratio_{ratio}')

path_fig_ker = os.path.join(path_fig_data,\
    f'kernels_bc_{num_data}_re_{id_repeat}')

path_fig_sample = os.path.join(path_fig_data, f'sample_{id_sample}')
print('>> Load result from  : ', path_fig_sample)
print('>> Load kernels from : ', path_fig_ker)

# ------------------------------------------------------------------------------
# load results of KLD
# ------------------------------------------------------------------------------
print('load kernels ...')
ker_init = io.imread(os.path.join(path_fig_ker, 'kernel_init.tif'))
ker_true = io.imread(os.path.join(path_fig_ker, 'kernel_true.tif'))
ker_FP   = io.imread(os.path.join(path_fig_ker, 'kernel_fp.tif'))
ker_BP   = io.imread(os.path.join(path_fig_ker, 'kernel_bp_ss.tif'))

# ------------------------------------------------------------------------------
print('load results (ss) ...')
y = io.imread(os.path.join(path_fig_sample, name_net, 'y.tif'))
x = io.imread(os.path.join(path_fig_sample, name_net, 'x.tif'))
y_pred_all = io.imread(os.path.join(path_fig_sample, name_net,\
    'y_pred_all.tif'))
y_pred = y_pred_all[num_iter_train]

# ------------------------------------------------------------------------------
Nz_kt, Ny_kt, Nx_kt = ker_true.shape
Nz_kf, Ny_kf, Nx_kf = ker_FP.shape
Nz_kb, Ny_kb, Nx_kb = ker_BP.shape
Nz, Ny, Nx = y.shape

# ------------------------------------------------------------------------------
vmax_img  = y.max()
vmax_diff = vmax_img

dict_ker         = {'cmap':'hot', 'vmin':0.0, 'vmax':ker_true.max()}
dict_ker_profile = {'linewidth': 0.5}
dict_img         = {'cmap': 'gray', 'vmin': 0.0, 'vmax': vmax_img}
dict_img_diff    = {'cmap': 'gray', 'vmin': 0.0, 'vmax': vmax_diff}

# ------------------------------------------------------------------------------
# s_fft = [318, 286, 286]
s_fft = y.shape
ker_true_fft = utils_data.fft_n(ker_true, s=s_fft) 
ker_FP_fft   = utils_data.fft_n(ker_FP, s=s_fft)

Nz_kf_ft, Ny_kf_ft, Nx_kf_ft = ker_FP_fft.shape

dict_kf_fft = {'cmap': 'hot', 'vmin': 0.0, 'vmax': ker_true_fft.max()}
dict_kf_fft_profile = {'linewidth': 0.5}

# ------------------------------------------------------------------------------
# load reaults of conventional methods
# ------------------------------------------------------------------------------
print('load results (others) ...')
data_gt, data_input = y, x
methods_name  = ['traditional', 'gaussian', 'butterworth', 'wiener_butterworth']
methods_color = ['#D04848', '#007F73', '#4CCD99', '#FFC700', '#FFF455']

def load_result(path, name, iter):
    out = []
    out.append(io.imread(os.path.join(path, name, f'deconv_{iter}.tif')))
    out.append(io.imread(os.path.join(path, name, 'deconv_bp.tif')))
    out.append(np.load(  os.path.join(path, name, f'deconv_metrics_{iter}.npy')))
    return out

# load results from conventional methods
out_trad = load_result(path_fig_sample, methods_name[0], methods_iter[0])
out_gaus = load_result(path_fig_sample, methods_name[1], methods_iter[1])
out_butt = load_result(path_fig_sample, methods_name[2], methods_iter[2])
out_wien = load_result(path_fig_sample, methods_name[3], methods_iter[3])

# ------------------------------------------------------------------------------
# show backward kernel image
# ------------------------------------------------------------------------------
print('plot backward kernel of different methods ...')
nr, nc = 3, 10
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4*nc, 2.4*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

def show(psf, ker_bp, axes, s=None, title=''):
    psf_fft  = utils_data.fft_n(psf, s=s)
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

print('sum of kf: ', ker_FP.sum())
print('sum of kb: ', ker_BP.sum())

plt.savefig(os.path.join(path_fig_ker, 'img_bp_ss.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_ker, 'img_bp_ss.svg'))

# ------------------------------------------------------------------------------
# plot FFT of backward kernels
# ------------------------------------------------------------------------------
print('plot fft of backward kernels ...')
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3.6*nc, 3.6*nr),\
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
    
plt.savefig(os.path.join(path_fig_ker, 'profile_bp_fft_ss'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_ker, 'profile_bp_fft_ss.svg'))

# ------------------------------------------------------------------------------
# show the restored images
# ------------------------------------------------------------------------------
print('plot restored images ...')
nr, nc = 4, 7
pos_text_x, pos_text_y = 5, 10
fontsize   = 'x-large'

line_start_xy = (38, 34)
line_end_xy   = (56, 19)
line_start_xz = (86, 44)
line_end_xz   = (86, 64)

id_slice   = Nz//2

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

dict_text = {'color': 'white', 'fontsize': fontsize}

axes[0,0].text(pos_text_x, pos_text_y, 'RAW (xy) ({:>.2f}, {:>.4f})'\
    .format(cal_psnr(data_input, data_gt), cal_ssim(data_input, data_gt)),\
    **dict_text)
axes[0,6].text(pos_text_x, pos_text_y, 'GT (xy)',   **dict_text)
axes[2,0].text(pos_text_x, pos_text_y, 'RAW (xz)',  **dict_text)
axes[2,6].text(pos_text_x, pos_text_y, 'GT (xz)',   **dict_text)

dict_line  = {'color': 'white', 'linewidth': 1}
dict_image = {'cmap': 'gray', 'vmin': 0, 'vmax': vmax_img}
dict_image_diff = {'cmap': 'gray', 'vmin': 0, 'vmax': vmax_diff}

axes[0,0].imshow(data_input[id_slice],    **dict_image)
axes[2,0].imshow(data_input[:, Ny//2, :], **dict_image)
print('RAW', 'PSNR:', cal_psnr(data_input, data_gt))

axes[0,0].plot((line_start_xy[0], line_end_xy[0]),\
    (line_start_xy[1], line_end_xy[1]),   **dict_line)
axes[2,0].plot((line_start_xz[0], line_end_xz[0]),\
    (line_start_xz[1], line_end_xz[1]),   **dict_line)

axes[0,6].imshow(data_gt[id_slice],    **dict_image)
axes[2,6].imshow(data_gt[:, Ny//2, :], **dict_image)

def show_result(out, axes, name):
    diff = np.abs(out[0]-data_gt)
    print(name, 'PSNR:', cal_psnr(out[0], data_gt))
    num_iter = out[-1].shape[0] - 1
    axes[0].text(pos_text_x, pos_text_y, '{} {:d} it'.format(name, num_iter),\
        **dict_text)
    axes[1].text(pos_text_x, pos_text_y, '({:>.2f}, {:>.4f})'\
        .format(out[-1][-1,0], out[-1][-1,1]), **dict_text)
    axes[0].imshow(out[0][id_slice],    **dict_image)
    axes[1].imshow(diff[id_slice],      **dict_image_diff)
    axes[2].imshow(out[0][:, Ny//2, :], **dict_image)
    axes[3].imshow(diff[:, Ny//2, :],   **dict_image_diff)

show_result(out_trad, axes[0:4,1], name='Traditional')
show_result(out_gaus, axes[0:4,2], name='Gaussian')
show_result(out_butt, axes[0:4,3], name='Butterworth')
show_result(out_wien, axes[0:4,4], name='WB')

axes[0,5].text(pos_text_x, pos_text_y, '{} {:d} it'\
    .format('KLD', num_iter_train), **dict_text)
axes[0,5].imshow(y_pred[id_slice], **dict_image)

print('KLD', 'PSNR:', cal_psnr(y_pred, data_gt))

axes[1,5].text(pos_text_x, pos_text_y, '({:>.2f}, {:>.4f})'\
    .format(cal_psnr(y_pred, y), cal_ssim(y_pred, y)), **dict_text)
axes[1,5].imshow(np.abs(y_pred-y)[id_slice], **dict_image_diff)

diff_ypred_y = np.abs(y_pred-y)
axes[2,5].imshow(y_pred[:, Ny//2, :], **dict_image)
axes[3,5].imshow(diff_ypred_y[:, Ny//2, :], **dict_image_diff)

io.imsave(fname=os.path.join(path_fig_sample, 'xz.tif'),\
    arr=y_pred[:, Ny//2, :], check_contrast=False)
io.imsave(fname=os.path.join(path_fig_sample, 'xy.tif'),\
    arr=y_pred[id_slice], check_contrast=False)

plt.savefig(os.path.join(path_fig_sample, 'img_restored_ss.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'img_restored_ss.svg'))

# profile line
# ------------------------------------------------------------------------------
# methods_name  = ['Traditional (30)', 'Gaussian (30)', 'Butterworth (30)',\
#     'WB (2)']
methods_name  = ['Traditional (2)', 'Gaussian (2)', 'Butterworth (2)',\
    'WB (2)']
colors = ['#B3BE9D', '#FDE767', '#F3B95F', '#E28154',]
# ['KernelNet', 'WB', 'Butterworth', 'Gaussian', 'Traditional', 'RAW']
# ['#D04848', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D','#6895D2']

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(6, 3),\
    constrained_layout=True)

dict_profile = {'linewidth': 0.5}

for ax in axes.ravel():
    for pos in ['top','bottom','left','right']:
        ax.spines[pos].set_linewidth(0.5)
        ax.tick_params(width=0.5)

for i, ax in enumerate(axes.ravel()):
    if i == 0:
        line_start = (line_start_xy[1],line_start_xy[0])
        line_end = (line_end_xy[1], line_end_xy[0])
        profile = profile_line(data_input[id_slice], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='RAW', color='#2A629A', **dict_profile)

        for out, name, color in zip([out_trad, out_gaus, out_butt, out_wien],\
            methods_name, colors):
            profile = profile_line(out[0][id_slice], line_start, line_end,\
                linewidth=1)
            ax.plot(profile, label=name, color=color, **dict_profile)

        profile = profile_line(y_pred[id_slice], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='KLD', color='red', **dict_profile)

        profile = profile_line(y[id_slice], line_start, line_end, linewidth=1)
        ax.plot(profile, label='GT', color='black', linestyle='--',\
            **dict_profile)

    if i == 1:
        line_start = (line_start_xz[1], line_start_xz[0])
        line_end = (line_end_xz[1], line_end_xz[0])
        profile = profile_line(data_input[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='RAW', color='#2A629A', **dict_profile)

        for out, name, color in zip([out_trad, out_gaus, out_butt, out_wien],\
            methods_name, colors):
            profile = profile_line(out[0][:, Ny//2, :], line_start, line_end,\
                linewidth=1)
            ax.plot(profile, label=name, color=color, **dict_profile)

        profile = profile_line(y_pred[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='KLD', color='red', **dict_profile)

        profile = profile_line(y[:, Ny//2, :], line_start, line_end,\
            linewidth=1)
        ax.plot(profile, label='GT', color='black', linestyle='--',\
            **dict_profile)

    ax.tick_params(direction='in')

    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    ax.set_ylabel('Intensity (AU)')
    ax.set_xlabel('Distance (pixel)')

plt.legend(fontsize='xx-small')
plt.savefig(os.path.join(path_fig_sample, 'img_restored_profile_ss.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'img_restored_profile_ss.svg'))

# ------------------------------------------------------------------------------
# show metrics curve
# ------------------------------------------------------------------------------
nr, nc = 1, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

psnrs, ssims, nccs = [], [], []

for i in range(len(y_pred_all)):
    psnrs.append(cal_psnr(y_pred_all[i], y))
    ssims.append(cal_ssim(y_pred_all[i], y))
    nccs.append(cal_ncc(y_pred_all[i], y))
mtrics_kld = np.stack([psnrs, ssims, nccs]).transpose()

axes[0].set_ylabel('PSNR')
axes[1].set_ylabel('SSIM')
axes[2].set_ylabel('NCC')

dict_line_metrics = {'linestyle': '-', 'marker':'.', 'markersize':2,\
    'linewidth':0.5}
dict_line_axh     = {'linestyle': '--', 'linewidth':0.5}

# methods_color = ['red', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D',\
#     '#6895D2']

for i in range(3):
    axes[i].plot(out_trad[-1][:,i], color=methods_color[4], label='Traditional',\
        **dict_line_metrics)
    axes[i].plot(out_gaus[-1][:,i], color=methods_color[3], label='Gaussian',\
        **dict_line_metrics)
    axes[i].plot(out_butt[-1][:,i], color=methods_color[2], label='Butterworth',\
        **dict_line_metrics)
    axes[i].plot(out_wien[-1][:,i], color=methods_color[1], label='WB',\
        **dict_line_metrics)
    axes[i].plot(mtrics_kld[:,i],  color=methods_color[0], label='KLD',\
        **dict_line_metrics)

    axes[i].axhline(y=mtrics_kld[num_iter_train,i],\
        color=methods_color[0], **dict_line_axh)
    axes[i].axhline(y=out_wien[-1][num_iter_train,i],\
        color=methods_color[1], **dict_line_axh)

    print(mtrics_kld[num_iter_train,i])
    print(out_wien[-1][num_iter_train,i])

    axes[i].spines[['right', 'top']].set_visible(False)
    axes[i].set_xlabel('Iteration Number')
    axes[i].set_xlim([0, None])
    axes[i].legend(edgecolor='white')

plt.savefig(os.path.join(path_fig_sample, 'curve_metrics_ss.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_fig_sample, 'curve_metrics_ss.svg'))
# ------------------------------------------------------------------------------
