import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva

cal_ssim= lambda x, y: eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
    multichannel=False, channle_axis=None, version_wang=False)
cal_mse = lambda x, y: eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())

# --------------------------------------------------------------------------------------
# dataset_name = 'SimuBeads3D_128'
dataset_name = 'SimuMix3D_128'
noise_level = 0
id_data = 1
scale_factor = 1

# --------------------------------------------------------------------------------------
# load data
# --------------------------------------------------------------------------------------
dataset_path    = os.path.join('F:', os.sep, 'Datasets', 'RLN', dataset_name)
data_gt_path    = os.path.join(dataset_path, 'gt')
data_input_path = os.path.join(dataset_path, f'input_noise_{noise_level}_sf_{scale_factor}_crop')

with open(os.path.join(dataset_path, 'test.txt')) as f:
    test_txt = f.read().splitlines() 

PSF        = io.imread(os.path.join(data_input_path, 'PSF.tif')).astype(np.float32)
data_gt    = io.imread(os.path.join(data_gt_path, test_txt[id_data])).astype(np.float32)
data_input = io.imread(os.path.join(data_input_path, test_txt[id_data])).astype(np.float32)

print('Load data from: ', dataset_path)
print('GT: {}, Input: {}, PSF: {}'.format(data_gt.shape, data_input.shape, PSF.shape))

Sx, Sy, Sz = data_gt.shape
PSF_align = dcv.align_size(PSF, Sx, Sy, Sz)
OTF_fp = np.fft.fftn(np.fft.ifftshift(dcv.align_size(PSF, Sx, Sy, Sz)))

# --------------------------------------------------------------------------------------
fig_path = os.path.join('outputs', 'figures', dataset_name.lower(),\
    f'scale_{scale_factor}_noise_{noise_level}', f'sample_{id_data}')
print('save figure to : ', fig_path)

out_trad    = io.imread(os.path.join(fig_path, 'traditional', 'deconv.tif'))
bp_trad     = io.imread(os.path.join(fig_path, 'traditional', 'deconv_bp.tif'))
bp_trad_otf = io.imread(os.path.join(fig_path, 'traditional', 'deconv_bp_otf.tif'))
out_trad_metrics = np.load(os.path.join(fig_path, 'traditional', 'deconv_metrics.npy'))

out_gaus    = io.imread(os.path.join(fig_path, 'gaussian', 'deconv.tif'))
bp_gaus     = io.imread(os.path.join(fig_path, 'gaussian', 'deconv_bp.tif'))
bp_gaus_otf = io.imread(os.path.join(fig_path, 'gaussian', 'deconv_bp_otf.tif'))
out_gaus_metrics = np.load(os.path.join(fig_path, 'gaussian', 'deconv_metrics.npy'))

out_bw    = io.imread(os.path.join(fig_path, 'butterworth', 'deconv.tif'))
bp_bw     = io.imread(os.path.join(fig_path, 'butterworth', 'deconv_bp.tif'))
bp_bw_otf = io.imread(os.path.join(fig_path, 'butterworth', 'deconv_bp_otf.tif'))
out_bw_metrics = np.load(os.path.join(fig_path, 'butterworth', 'deconv_metrics.npy'))

out_wb    = io.imread(os.path.join(fig_path, 'wiener_butterworth', 'deconv.tif'))
bp_wb     = io.imread(os.path.join(fig_path, 'wiener_butterworth', 'deconv_bp.tif'))
bp_wb_otf = io.imread(os.path.join(fig_path, 'wiener_butterworth', 'deconv_bp_otf.tif'))
out_wb_metrics = np.load(os.path.join(fig_path, 'wiener_butterworth', 'deconv_metrics.npy'))

# --------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------
nr, nc = 4, 5
vmax_gt   = data_gt.max() * 0.4
vmax_diff = vmax_gt
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(2.4 * nc, 2.4 * nr),\
    constrained_layout=True)

axes[0,0].set_title('GT (xy)')
axes[0,0].imshow(data_gt[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt)
axes[1,0].set_title('RAW (xy)')
axes[1,0].imshow(data_input[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt)
axes[2,0].set_title('GT (xz)')
axes[2,0].imshow(data_gt[:, Sy//2, :], cmap='gray', vmin=0, vmax=vmax_gt)
axes[3,0].set_title('RAW (xz)')
axes[3,0].imshow(data_input[:, Sy//2, :], cmap='gray', vmin=0, vmax=vmax_gt)

def show_res_diff_xy(out, axes, name, num_iter):
    num_iter = num_iter - 1
    axes[0].set_title('{} {:d} it'.format(name, num_iter))
    axes[0].imshow(out[Sx//2], cmap='gray', vmin=0, vmax=vmax_gt)
    axes[1].imshow(np.abs(out-data_gt)[Sx//2], cmap='gray', vmin=0, vmax=vmax_diff)
    axes[1].set_title('[{:>.2f}, {:>.4f}]'.format(cal_mse(out, data_gt), cal_ssim(out, data_gt)))

def show_res_diff_xz(out, axes):
    axes[0].imshow(out[:, Sy//2, :], cmap='gray', vmin=0, vmax=vmax_gt)
    axes[1].imshow(np.abs(out-data_gt)[:, Sy//2, :], cmap='gray', vmin=0, vmax=vmax_diff)

show_res_diff_xy(out_trad, axes[0:2,1], name='Traditional', num_iter=out_trad_metrics.shape[0])
show_res_diff_xy(out_gaus, axes[0:2,2], name='Gaussian', num_iter=out_gaus_metrics.shape[0])
show_res_diff_xy(out_bw, axes[0:2,3], name='Butterworth', num_iter=out_bw_metrics.shape[0])
show_res_diff_xy(out_wb, axes[0:2,4], name='WB', num_iter=out_wb_metrics.shape[0])

show_res_diff_xz(out_trad, axes[2:4,1])
show_res_diff_xz(out_gaus, axes[2:4,2])
show_res_diff_xz(out_bw, axes[2:4,3])
show_res_diff_xz(out_wb, axes[2:4,4])

plt.savefig(os.path.join(fig_path, 'img_restored_conventional.png'))

# --------------------------------------------------------------------------------------
# show OTF
# --------------------------------------------------------------------------------------
def plot_curve(x_ft, axes, name, color):
    x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
    x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * OTF_fp))
    axes[0].plot(x_ft_shift_abs[Sx//2, Sy//2:, Sz//2], label=name, color=color)
    axes[1].plot(x_ft_x_psf_ft_shift_abs[Sx//2, Sy//2:, Sz//2], label=name, color=color)

nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(4.0 * nc, 4.0* nr), constrained_layout=True)
axes[0].set_title('|FT(BP)|')
axes[1].set_title('|FT(BP) x FT(PSF)|')
plot_curve(bp_trad_otf, [axes[0], axes[1]], name='traditional', color='blue')
plot_curve(bp_gaus_otf, [axes[0], axes[1]], name='gaussian', color='cyan')
plot_curve(bp_bw_otf,   [axes[0], axes[1]], name='butterworth', color='orange')
# plot_curve(bp_wiener_otf, [axes[1], axes[2]], name='wiener', color='green')
plot_curve(bp_wb_otf, [axes[0], axes[1]], name='wiener-butterworth', color='orangered')
axes[0].legend()
axes[1].legend()
for ax in [axes[0], axes[1]]:
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Normalized value')

plt.savefig(os.path.join(fig_path, 'profile_bp_fft_conventional.png'))

# --------------------------------------------------------------------------------------
# show metrics curve
# --------------------------------------------------------------------------------------
nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(4.0 * nc, 4.0 * nr), constrained_layout=True)
axes[0].set_title('PSNR')
axes[1].set_title('SSIM')

for i in range(2):
    axes[i].plot(out_trad_metrics[:,i], linestyle='-', marker='.', color='blue',    label='Traditional')
    axes[i].plot(out_gaus_metrics[:,i], linestyle='-', marker='.', color='green',   label='Gaussian')
    axes[i].plot(out_bw_metrics[:,i],   linestyle='-', marker='.', color='orange',  label='Butterworth')
    axes[i].plot(out_wb_metrics[:,i],   linestyle='-', marker='.', color='orangered', label='Wiener-Butterworth')
    axes[i].set_xlabel('Iteration Number')
    axes[i].legend()
    axes[i].set_xlim([0, None])

plt.savefig(os.path.join(fig_path, 'curve_metrics_conventional.png'))