import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
from skimage.measure import profile_line
import numpy as np
import os
from utils import evaluation as eva
import matplotlib.patches as patches
import matplotlib.colors as colors
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
def cal_ssim(x, y):
    return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
        version_wang=False)

def cal_psnr(x, y):
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())

# ------------------------------------------------------------------------------
dataset_name_test, id_data = 'F-actin_Nonlinear', 5
# dataset_name_test, id_data = 'Microtubules2', 0

scale_factor, std_gauss = 1, 9
name_net = 'kernelnet'
eps = 0.000001

path_result = os.path.join('outputs', 'figures', dataset_name_test,\
    f'scale_{scale_factor}_gauss_{std_gauss}_poiss_1_ratio_1')
path_result_sample = os.path.join(path_result, f'sample_{id_data}')
path_kernel = os.path.join(path_result, 'kernels_bc_1_re_1')

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print('-'*80)
print('load results from :', path_result_sample)
print('load kernels from :', path_kernel)

ker_init = io.imread(os.path.join(path_kernel, 'kernel_init.tif'))
ker_true = io.imread(os.path.join(path_kernel, 'kernel_true.tif'))
ker_FP   = io.imread(os.path.join(path_kernel, 'kernel_fp.tif'))
ker_BP   = io.imread(os.path.join(path_kernel, 'kernel_bp.tif'))

y      = io.imread(os.path.join(path_result_sample, name_net, 'y.tif'))
x      = io.imread(os.path.join(path_result_sample, name_net, 'x.tif'))
x0     = io.imread(os.path.join(path_result_sample, name_net, 'x0.tif'))
y_fp   = io.imread(os.path.join(path_result_sample, name_net, 'y_fp.tif'))
x0_fp  = io.imread(os.path.join(path_result_sample, name_net, 'x0_fp.tif'))
bp     = io.imread(os.path.join(path_result_sample, name_net, 'bp.tif'))
y_pred = io.imread(os.path.join(path_result_sample, name_net, 'y_pred_all.tif'))

# the imread funciton will automaticly reshape the data when having 3 channels.
if y_pred.shape[-1] in [3, 4]: y_pred = np.transpose(y_pred, axes=(-1, 0, 1))
y_pred = y_pred[2]

Ny_kt, Nx_kt = ker_true.shape
Ny_kf, Nx_kf = ker_FP.shape
Ny_kb, Nx_kb = ker_BP.shape
Ny_i,  Nx_i  = y.shape

# ------------------------------------------------------------------------------
vmax_ker, color_map_ker = ker_FP.max(), 'hot'
vmax_img, color_map_img = y.max() * 0.6, 'gray'
vmax_diff = vmax_img

# ------------------------------------------------------------------------------
# Show forward kernel image
# ------------------------------------------------------------------------------
print('plot forward kernels ...')
nr, nc = 1, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes[0:2].ravel(): ax.set_axis_off()

dict_ker = {'vmin':0.0, 'vmax':vmax_ker, 'cmap':'hot'}

axes[0].imshow(ker_true, **dict_ker)      
axes[1].imshow(ker_FP,   **dict_ker)       
axes[2].plot(ker_true[Ny_kt//2], color='black', label='True')
axes[2].plot(ker_init[Ny_kt//2], color='blue',  label='init')
axes[2].plot(ker_FP  [Ny_kf//2], color='red',   label=name_net) 

axes[0].set_title('PSF (true) [' + str(np.round(ker_true.sum(), 4)) +']')
axes[1].set_title(f'PSF ({name_net}) [' + str(np.round(ker_FP.sum(), 4)) +']')
axes[2].set_title('PSF profile')
axes[2].set_xlim([0, None])
axes[2].set_ylim([0, None])
axes[2].legend()

plt.savefig(os.path.join(path_kernel, 'img_fp'))

# ------------------------------------------------------------------------------
# show forward intermediate results
# ------------------------------------------------------------------------------
print('plot forward intermediate results ...')
nr, nc = 2, 5
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

dict_img = {'cmap':color_map_img, 'vmin':0.0, 'vmax':vmax_img}
dict_img_diff = {'cmap':'gray', 'vmin':0.0, 'vmax':vmax_diff}

axes[0,0].imshow(y,  **dict_img)
axes[0,1].imshow(x0, **dict_img)
axes[1,0].imshow(x,  **dict_img)

axes[0,0].set_title('HR (xy)')
axes[0,1].set_title('x0 ({:.2f})'.format(cal_psnr(x0, y)))
axes[1,0].set_title('LR (xy)')

axes[0,2].imshow(ker_FP, **dict_ker)
axes[0,3].imshow(x0_fp,  **dict_img)
axes[0,4].imshow(y_fp,   **dict_img)
axes[1,1].imshow(np.abs(x0-y),   **dict_img_diff)
axes[1,2].imshow(ker_true,       **dict_ker)
axes[1,4].imshow(np.abs(y_fp-x), **dict_img_diff)

axes[0,2].set_title(f'PSF ({name_net})')
axes[0,3].set_title('FP(x0)')
axes[0,4].set_title('FP(HR)')
axes[1,1].set_title('|x0-HR|')
axes[1,2].set_title('PSF (true)')
axes[1,4].set_title('|FP(HR)-LR| ({:.2f})'.format(cal_psnr(y_fp, x)))

plt.savefig(os.path.join(path_result_sample, 'img_fp_inter'))

# ------------------------------------------------------------------------------
# show FFT of the forward kernel
# ------------------------------------------------------------------------------
print('plot fft of forward kernel ...')
ker_true_fft = utils_data.fft_n(ker_true, s=y.shape) 
ker_FP_fft   = utils_data.fft_n(ker_FP,   s=y.shape)
Ny_ft, Nx_ft = ker_FP_fft.shape

# ------------------------------------------------------------------------------
nr, nc = 1, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes[0:2].ravel(): ax.set_axis_off()

axes[0].set_title('FT(PSF (true))')
axes[1].set_title('FT(PSF (learned))')

dict_ker_fft = {'cmap':'hot', 'vmin':0.0, 'vmax':ker_FP_fft.max()}

axes[0].imshow(ker_true_fft, **dict_ker_fft)
axes[1].imshow(ker_FP_fft,   **dict_ker_fft)   
axes[2].plot(ker_true_fft[Ny_ft//2], color='black', label='True')
axes[2].plot(ker_FP_fft[Ny_ft//2],   color='red', label=name_net)
axes[2].legend()

plt.savefig(os.path.join(path_kernel, 'img_fp_fft'))

# ------------------------------------------------------------------------------
# show backward intermediate results
# ------------------------------------------------------------------------------
print('plot backward intermediate results ...')
nr, nc = 2, 6
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,0].imshow(y, **dict_img)
axes[1,0].imshow(x, **dict_img)
axes[0,0].set_title('HR (xy)')
axes[1,0].set_title('LR (xy) ({:.2f}, {:.4f})'\
    .format(cal_psnr(x, y), cal_ssim(x, y)))

axes[0,1].imshow(x0_fp, **dict_img)
axes[1,1].imshow(np.abs(x0-y), **dict_img_diff)
axes[0,1].set_title('FP(x0)')
axes[1,1].set_title('|x0-HR|')

axes[0,2].imshow((x/(x0_fp + eps)), cmap='seismic', vmin=0.5, vmax=1.5)
axes[1,2].imshow(np.abs(y_fp-x),  **dict_img_diff)
axes[0,2].set_title('LR/FP(x0)')
axes[1,2].set_title('|FP(HR)-LR|')

axes[0,3].imshow(ker_BP, cmap=color_map_ker, vmin=0.0, vmax=np.max(ker_BP))
axes[0,3].set_title('BP kernel [{:.2f}]'.format(ker_BP.sum()))
axes[1,3].hlines(0, xmin=0, xmax=Nx_kb, color='black')
axes[1,3].plot(ker_BP[Ny_kb//2], color='red')
axes[1,3].set_axis_on()

axes[0,4].imshow(bp, cmap='seismic', vmin=0.0, vmax=2.0)
axes[0,4].set_title('BP(LR/FP(x0))')

axes[0,5].imshow(y_pred, **dict_img)
axes[1,5].imshow(np.abs(y_pred-y), **dict_img_diff)
axes[0,5].set_title('xk ({:.4f})'.format(cal_ssim(y_pred, y)))
axes[1,5].set_title('|xk-HR| ({:.2f})'.format(cal_psnr(y_pred, y)))

plt.savefig(os.path.join(path_result_sample, 'img_bp_inter'))

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
print('load restored images ...')
y_bd = io.imread(os.path.join(path_result_sample, 'deconvblind',\
    'deconv.tif')).astype(np.float32)
y_trad = io.imread(os.path.join(path_result_sample, 'traditional',\
    'deconv.tif')).astype(np.float32)
data = [x, y_bd, y_trad, y_pred, y]

N_meth = len(data) # number of methods (include raw and gt)
methods_name = ['WF', 'DeconvBlind', 'RLD#', 'KLD', 'SIM']

# ------------------------------------------------------------------------------
if dataset_name_test == 'F-actin_Nonlinear':
    pos, size = [200,200], [100,200]
    color_map_img = 'hot'

if dataset_name_test == 'Microtubules2':
    pos, size = [200,200], [100,200]
    color_map_img = colors.LinearSegmentedColormap.from_list("", \
        ["black", "#03AED2", "white"])

vmax_img = data[-1].max()*0.6
# ------------------------------------------------------------------------------
# whole image
nr, nc = 2, N_meth
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

for ax in axes.ravel():
    ax.set_axis_off()

dict_img = {'cmap':color_map_img, 'vmin': 0.0, 'vmax':vmax_img}

for i in range(N_meth):
    img = data[i]
    axes[0,i].imshow(img, **dict_img)
    box = patches.Rectangle(xy=(pos[1],pos[0]), width=size[1], height=size[0],\
        fill=False, edgecolor='white', linewidth=0.5)
    axes[0,i].add_patch(box)

    axes[1,i].imshow(img[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]],\
        **dict_img)

plt.savefig(os.path.join(path_result_sample, 'image_restored.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_result_sample, 'image_restored.svg'))

print('-'*80)
print('PSNR | SSIM:')
for i in range(N_meth-1):
    print(cal_psnr(data[i], data[-1]), cal_ssim(data[i], data[-1]))
print('-'*80)

# ------------------------------------------------------------------------------
# profile line
fig, axes = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(6, 3),\
    constrained_layout=True)

for i in range(N_meth):
    start, end=(35,0), (35,100)
    profile = profile_line(data[i],start,end, linewidth=2)
    axes.plot(profile, label=methods_name[i])

plt.legend()
plt.savefig(os.path.join(path_result_sample, 'image_restored_profile.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_result_sample, 'image_restored_profile.svg'))
