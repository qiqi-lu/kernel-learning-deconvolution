import numpy as np
import torch, tqdm, os, sys
import matplotlib.pyplot as plt
from torch import nn
from utils import device_utils, dataset_utils
from utils import evaluation as eva
from torchvision import transforms

from models import rlsr, rrdbnet, rln

# -----------------------------------------------------------------------------------
# device = device_utils.get_device(gpu_id=5)    # when use local server
# device = torch.device("cuda")                   # when use cloud server
device = torch.device("cpu")                  # when use cpu

# -----------------------------------------------------------------------------------
# Choose data set
# data_set_name = 'tinymicro_synth'
# data_set_name = 'tinymicro_real'
data_set_name = 'lung3_synth'
# data_set_name = 'biosr_real'
# data_set_name = 'msi_synth'
id_data = 100
# -----------------------------------------------------------------------------------
fig_dir = os.path.join('outputs', 'figures', data_set_name)
if os.path.exists(fig_dir) == False: os.makedirs(fig_dir, exist_ok=True)

# -----------------------------------------------------------------------------------
data_transform = transforms.ToTensor()

# -----------------------------------------------------------------------------------
if data_set_name == 'tinymicro_synth':
    # TinyMicro (synth)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization = (False, False)
    in_channels = 3
    init_mode = 'bicubic'
    data_range = 255

if data_set_name == 'tinymicro_real':
    # TinyMicro (real)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization = (False, False)
    in_channels = 3
    init_mode = 'net'
    data_range = 255

if data_set_name == 'biosr_real':
    pass
if data_set_name == 'lung3_synth':
    # Lung3 (synth)
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'
    hr_root_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939')
    lr_root_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')

    hr_txt_file_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'lr.txt') 
    normalization = (False, True)
    in_channels, scale_factor = 1, 4
    # init_mode = 'bicubic'
    init_mode = 'ave_bicubic'
    data_range = 1.0

if data_set_name == 'msi_synth':
    pass

test_data = dataset_utils.SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
        hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
        transform=data_transform, normalization=normalization)

# -----------------------------------------------------------------------------------
print('Model construction ...',end=' ')
num_iter, num_block, n_features, backbone_type  = 2, (1,3), 8, 'rrdb'
use_prior, RL_version, fp_mode, batchnorm = False, 'RL', 'pre-trained', True
forw_proj = None
model_dict = {}
model_dict['name']  = 'RLSR'
# model_dict['model'] = rlsr.RLSR(scale=4, in_channels=3, n_features=16, n_blocks=num_block, n_iter=num_iter,\
#         multi_output=False, bias=True, inter=True, init_mode='bilinear',pm='zeros').to(device)
if fp_mode == 'pre-trained':
    model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_16_lr_0.001_1_8_bin_ave',\
        'epoch_20000.pt')
    model_para = torch.load(model_path, map_location=device)
    backbone = rrdbnet.backbone(in_channels=in_channels, n_features=8,\
        n_blocks=1, growth_channels=8 // 2, bias=True, pm='zeros')
    forw_proj = rlsr.ForwardProject(backbone=backbone, in_channels=in_channels,\
        n_features=8, scale=scale_factor, bias=True, bn=False, pm='zeros',\
        kernel_size=3, only_net=False, pixel_binning_mode='ave').to(device)
    forw_proj.load_state_dict(model_para['model_state_dict'])
    forw_proj.eval()

if fp_mode == 'known':
    ks = 25
    ker = rln.gauss_kernel_2d(shape=(ks, ks), sigma=2.0)
    ker = ker.repeat(repeats=(in_channels, 1, 1, 1)).to(device=device)
    padd = lambda x: torch.nn.functional.pad(input=x, pad=(ks//2, ks//2, ks//2, ks//2), mode='reflect')
    conv = lambda x: torch.nn.functional.conv2d(input=padd(x), weight=ker, stride=1, groups=in_channels)
    forw_proj = lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor)

model_dict['model'] = rlsr.RLSR(img_size=(128, 128), scale=scale_factor, in_channels=in_channels,\
    n_features=n_features, n_blocks=num_block, n_iter=num_iter, bias=True,\
    init_mode=init_mode, pm='zeros', backbone_type=backbone_type, window_size=8,\
    mlp_ratio=2, kernel_size=3, only_net=False, output_mode='inters', use_prior=use_prior,\
    RL_version=RL_version, pixel_binning_mode='ave', model_mode='div_mul', constraint_01=True,\
    forw_proj=forw_proj, bn=batchnorm, cat_x=False).to(device)

if data_set_name == 'tinymicro_synth':
    model_dict['epoch'] = '95000'
    model_dict['model_ver'] = 'rlsr_bs_2_lr_0.001_iter_2_block_(5,5)_feature_16_mul_ig_1.0_rrdb_bicubic_ay_alterupdate_direct_sigmoid'

if data_set_name == 'tinymicro_real':
    model_dict['epoch'] = '95000'
    model_dict['model_ver'] = 'rlsr_bs_3_lr_0.001_iter_1_block_(5,5)_feature_16_mul_ig_1.0_rrdb_net_ay_alterupdate_direct_sigmoid'

if data_set_name == 'lung3_synth':
    model_dict['epoch'] = '95000'
    # model_dict['model_ver'] = 'rlsr_bs_3_lr_0.001_iter_2_block_(1,2)_feature_16_mul_ig_1.0_rrdb_bicubic_ay_alterupdate_direct_sigmoid_ayb'
    # model_dict['model_ver'] = 'rlsr_bs_4_lr_0.001_iter_2_block_(1,3)_feature_8_1.0_rrdb_ave_bicubic_ay_ayb_RL_prior'
    model_dict['model_ver'] = 'rlsr_bs_4_lr_0.01_iter_1_block_(1,3)_feature_8_1.0_rrdb_ave_bicubic_ay_RL_bin_ave_div_mul_fp_mode_pre-trained_bn_one'

print('(done)')

# -----------------------------------------------------------------------------------
model_path = os.path.join('checkpoints', data_set_name, model_dict['model_ver'],\
    'epoch_{}.pt'.format(model_dict['epoch']))
print('Model: ', model_path, end=' ')
model_para = torch.load(model_path, map_location=device) # load weights
model = model_dict['model'] # construct model
model.load_state_dict(model_para['model_state_dict'])
model.eval()

# -----------------------------------------------------------------------------------
interp = lambda x: torch.nn.functional.interpolate(x, scale_factor=4, mode='nearest')
to_npy = lambda x: x.cpu().detach().numpy().transpose((0, 2, 3, 1))
def to_img(x):
    x = (x * data_range).cpu().detach().numpy().transpose((0, 2, 3, 1))
    if in_channels == 3: x = x.astype(np.uint8)
    return x

colorbar = lambda fig, img, axe: fig.colorbar(img, ax=axe, aspect=25, shrink=0.6)

# ------------------------------------------------------------------------------------
ds = test_data[id_data] # get singel sample

x = torch.unsqueeze(ds['lr'], 0).to(device)
y = torch.unsqueeze(ds['hr'], 0).to(device)

# LR as input
y_out = model(x) # [fp, dv, bp, xsr]

# GT as input
y_fp = model.forward_project(y)
x_dv_y_fp = torch.div(x, torch.add(y_fp, 0.00001))

if RL_version == 'RL': y_bp = model.back_project(x_dv_y_fp)
if RL_version == 'ISRA': y_bp = model.back_project(x)/(model.back_project(y_fp) + 0.00001)

if use_prior == True: y_prior = model.cal_prior(y)

# ------------------------------------------------------------------------------------
# convert to numpy data
x, y = to_img(interp(x)), to_img(y)
y_fp = to_img(interp(y_fp))
y_bp = to_npy(y_bp)
if use_prior == True: y_prior = to_npy(y_prior)

init_img = to_img(y_out[0])
y_pred = y_out[1:]

# ------------------------------------------------------------------------------------
# Show real image
# ------------------------------------------------------------------------------------
if use_prior == True:  num_inter_out = 6
if use_prior == False: num_inter_out = 4

num_iter  = int(len(y_pred) // num_inter_out)
nr, nc    = np.maximum(num_iter, 2) + 1, num_inter_out + 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=600, figsize=(2.4 * nc, 2.4 * nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
id_channel_show = 0
# ------------------------------------------------------------------------------------
# show LR, and initial image.
if in_channels == 3: 
    img_lr   = axes[0,0].imshow(x[0]) # LR image
    img_init = axes[1,0].imshow(init_img[0]) # init image
if in_channels == 1:
    img_lr   = axes[0,0].imshow(x[0], cmap='gray', vmin=0.0, vmax=0.6) # LR image
    img_init = axes[1,0].imshow(init_img[0], cmap='gray', vmin=0.0, vmax=0.6) # init image

axes[0,0].set_title('LR')
axes[1,0].set_title('init ({:.4f}|{:.2f})'.format(*eva.measure(y, init_img, data_range=data_range)))

colorbar(fig, img_lr, axes[0,0])
colorbar(fig, img_init, axes[1,0])
# ------------------------------------------------------------------------------------
for i in range(num_iter):
    fp = to_img(interp(y_pred[0 + i*num_inter_out]))
    dv = to_npy(interp(y_pred[1 + i*num_inter_out]))
    bp = to_npy(y_pred[2 + i*num_inter_out])
    o  = to_img(y_pred[3 + i*num_inter_out])
    if use_prior == True:
        o0 = to_img(y_pred[4 + i*num_inter_out])
        op = to_npy(y_pred[5 + i*num_inter_out])

    if in_channels == 3: 
        img1 = axes[i,1].imshow(fp[0])
        img4 = axes[i,4].imshow(o[0])
        if use_prior == True: 
            img5 = axes[i,5].imshow(o0[0])
    if in_channels == 1: 
        img1 = axes[i,1].imshow(fp[0], cmap='gray', vmin=0.0, vmax=0.6)
        img4 = axes[i,4].imshow(o[0], cmap='gray', vmin=0.0, vmax=0.6)
        if use_prior == True:
            img5 = axes[i,5].imshow(o0[0], cmap='gray', vmin=0.0, vmax=0.6)

    img2 = axes[i,2].imshow(dv[0, ..., id_channel_show], cmap='seismic', vmin=0.0, vmax=2.0)
    img3 = axes[i,3].imshow(bp[0, ..., id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2)

    axes[i,1].set_title('FP_' + str(i+1))
    axes[i,2].set_title('DV_' + str(i+1))
    axes[i,3].set_title('BP_' + str(i+1))
    axes[i,4].set_title('x_{} ({:.4f}|{:.2f})'.format(str(i+1), *eva.measure(y, o, data_range=data_range)))

    colorbar(fig, img1, axes[i,1])
    colorbar(fig, img2, axes[i,2])
    colorbar(fig, img3, axes[i,3])
    colorbar(fig, img4, axes[i,4])

    if use_prior:
        axes[i,5].set_title('x0_{} ({:.4f}|{:.2f})'.format(str(i+1), *eva.measure(y, o0, data_range=data_range)))
        axes[i,6].set_title('prior_' + str(i+1))
        # img6 = axes[i,6].imshow(op[0, ..., id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2)
        img6 = axes[i,6].imshow(op[0, ..., id_channel_show], cmap='seismic', vmin=-0.2, vmax=0.2)
        colorbar(fig, img5, axes[i,5])
        colorbar(fig, img6, axes[i,6])
# ------------------------------------------------------------------------------------
# The results using GT as input
if in_channels == 3:
    img_hr    = axes[-1,0].imshow(y[0])     # HR image
    img_hr_fp = axes[-1,1].imshow(y_fp[0])  # image after forward project
    img_hr_up = axes[-1,4].imshow((y_bp[0] * y[0]).astype(np.uint8)) # HR image after update
    if use_prior:
        img_hr_up_p = axes[-1,6].imshow((y_bp[0] * y[0] + (y_prior[0] + 0.0001)).astype(np.uint8))
if in_channels == 1:
    img_hr    = axes[-1,0].imshow(y[0], cmap='gray', vmin=0.0, vmax=0.6)
    img_hr_fp = axes[-1,1].imshow(y_fp[0], cmap='gray', vmin=0.0, vmax=0.6)
    img_hr_up = axes[-1,4].imshow(y_bp[0] * y[0], cmap='gray', vmin=0.0, vmax=0.6)
    if use_prior:
        img_hr_up_p = axes[-1,6].imshow(y_bp[0] * y[0] + (y_prior[0] + 0.0001), cmap='gray', vmin=0.0, vmax=0.6)

img_hr_dv = axes[-1,2].imshow(x[0, ..., id_channel_show] / (y_fp[0, ..., id_channel_show] + 0.0001),
    cmap='seismic', vmin=0.0, vmax=2.0) # divide
img_hr_bp = axes[-1,3].imshow(y_bp[0, ..., id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2)

if use_prior == True:
    # img_hr_prior = axes[-1,5].imshow(y_prior[0, ..., id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2) 
    img_hr_prior = axes[-1,5].imshow(y_prior[0, ..., id_channel_show], cmap='seismic', vmin=-0.2, vmax=0.2) 
    
axes[-1,0].set_title('HR')
axes[-1,1].set_title('HR-FP')
axes[-1,2].set_title('HR-FP-DV')
axes[-1,3].set_title('HR-FP-DV-BP')
axes[-1,4].set_title('HR-up ({:.4f}|{:.2f})'.format(*eva.measure(y, y_bp * y, data_range=data_range)))
if use_prior:
    axes[-1,5].set_title('HR-prior')
    axes[-1,6].set_title('HR-up ({:.4f}|{:.2f})'.format(*eva.measure(y, y_bp * y + (y_prior + 0.0001), data_range=data_range)))


colorbar(fig, img_hr, axes[-1,0])
colorbar(fig, img_hr_fp, axes[-1,1])
colorbar(fig, img_hr_dv, axes[-1,2])
colorbar(fig, img_hr_bp, axes[-1,3])
colorbar(fig, img_hr_up, axes[-1,4])
if use_prior:
    colorbar(fig, img_hr_prior, axes[-1,5])
    colorbar(fig, img_hr_up_p, axes[-1,6])

plt.savefig(os.path.join(fig_dir,'comparison_inter_'+str(id_data)))

# ------------------------------------------------------------------------------------
# show single channel
# ------------------------------------------------------------------------------------
if in_channels == 1:
    id_channel_show =0
else:
    id_channel_show = 1

nr, nc = np.maximum(num_iter, 2) + 1, 5
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=600, figsize=(2.4 * nc, 2.4 * nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# ------------------------------------------------------------------------------------
# low resolution image (input) and initialization
img_lr   = axes[0,0].imshow(x[0, ..., id_channel_show], cmap='gray', vmin=0.0, vmax=data_range) 
img_init = axes[1,0].imshow(init_img[0, ..., id_channel_show], cmap='gray', vmin=0.0, vmax=data_range) 
axes[0,0].set_title('LR')
axes[1,0].set_title('init')
colorbar(fig, img_lr, axes[0,0])
colorbar(fig, img_init, axes[1,0])

# intermediate outputs
for i in range(num_iter):
    fp = to_img(interp(y_pred[0 + i*4]))
    dv = to_npy(interp(y_pred[1 + i*4]))
    bp = to_npy(y_pred[2 + i*4])
    o  = to_img(y_pred[3 + i*4])

    img1 = axes[i,1].imshow(fp[0,...,id_channel_show], cmap='gray', vmin=0.0, vmax=data_range)
    img2 = axes[i,2].imshow(dv[0,...,id_channel_show], cmap='seismic', vmin=0.0, vmax=2.0)
    img3 = axes[i,3].imshow(bp[0,...,id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2)
    img4 = axes[i,4].imshow( o[0,...,id_channel_show], cmap='gray', vmin=0.0, vmax=data_range)

    axes[i,1].set_title('FP_{}'.format(i+1))
    axes[i,2].set_title('DV_{}'.format(i+1))
    axes[i,3].set_title('BP_{}'.format(i+1))
    axes[i,4].set_title('x_{}'.format(i+1))

    colorbar(fig, img1, axes[i,1])
    colorbar(fig, img2, axes[i,2])
    colorbar(fig, img3, axes[i,3])
    colorbar(fig, img4, axes[i,4])

# The results using GT as input
img_hr    = axes[-1,0].imshow(y[0, ..., id_channel_show], cmap='gray', vmin=0.0, vmax=data_range)                 # high resolution image (gt)
img_hr_fp = axes[-1,1].imshow(y_fp[0, ..., id_channel_show], cmap='gray', vmin=0.0, vmax=data_range)              # high resolution after forward project
img_hr_dv = axes[-1,2].imshow(x[0, ..., id_channel_show] / (y_fp[0, ..., id_channel_show] + 0.00001),\
    cmap='seismic', vmin=0.0, vmax=2.0) # divide
img_hr_bp = axes[-1,3].imshow(y_bp[0, ..., id_channel_show], cmap='seismic', vmin=0.8, vmax=1.2) 
img_hr_up = axes[-1,4].imshow(y_bp[0, ..., id_channel_show] * y[0, ..., id_channel_show],\
    cmap='gray', vmin=0.0, vmax=data_range)   # high resolution after update
axes[-1,0].set_title('HR')
axes[-1,1].set_title('HR-FP')
axes[-1,2].set_title('HR-FP-DV')
axes[-1,3].set_title('HR-FP-DV-BP')
axes[-1,4].set_title('HR-up')

colorbar(fig,img_hr,axes[-1,0])
colorbar(fig,img_hr_fp,axes[-1,1])
colorbar(fig,img_hr_dv,axes[-1,2])
colorbar(fig,img_hr_bp,axes[-1,3])
colorbar(fig,img_hr_up,axes[-1,4])

plt.savefig(os.path.join(fig_dir,'comparison_inter_G_'+str(id_data)))

# ------------------------------------------------------------------------------------
hist = lambda ax, img, range=(0.0, 1.0): ax.hist(img.flatten(), bins=255, range=range)

fig_hist, axes_hist = plt.subplots(nrows=nr - 1, ncols=nc ,dpi=300, figsize=(2.4 * nc, 2.4 * (nr - 1)),\
                                   constrained_layout=True)

hist(axes_hist[0,0], x[0, ..., id_channel_show], range=(0.0, data_range))
hist(axes_hist[1,0], y[0, ..., id_channel_show], range=(0.0, data_range))
axes_hist[0,0].set_title('LR')
axes_hist[1,0].set_title('HR')

for i in range(num_iter):
    fp = to_img(interp(y_pred[0 + i*4]))
    dv = to_npy(interp(y_pred[1 + i*4]))
    bp = to_npy(y_pred[2 + i*4])
    o  = to_img(y_pred[3 + i*4])

    hist(ax=axes_hist[i,1], img=fp[0, ..., id_channel_show], range=(0.0, data_range))
    hist(ax=axes_hist[i,2], img=dv[0, ..., id_channel_show], range=(0.0, 2.0))
    hist(ax=axes_hist[i,3], img=bp[0, ..., id_channel_show], range=(0.8, 1.2))
    hist(ax=axes_hist[i,4], img= o[0, ..., id_channel_show], range=(0.0, data_range))

    axes_hist[i,1].set_title('FP_' + str(i+1))
    axes_hist[i,2].set_title('DV_' + str(i+1))
    axes_hist[i,3].set_title('BP_' + str(i+1))
    axes_hist[i,4].set_title('x_' + str(i+1))

    for j in range(nc): axes_hist[i, j].set_ylim([0, 10000])

plt.savefig(os.path.join(fig_dir, 'comparison_inter_G_hist' + str(id_data)))

