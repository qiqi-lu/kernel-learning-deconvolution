import numpy as np
import torch, tqdm, os, time, sys
from torch.utils.data import DataLoader
from utils import dataset_utils
from utils import evaluation as eva
import matplotlib.pyplot as plt
from torchvision import transforms

from models import edsr, rln, rcan, swinir, rlsr, srcnn, srresnet, rrdbnet, dfcan
import utils.image_process as ip
import methods.rld as rld
# -----------------------------------------------------------------------------------
if sys.platform == 'linux' or sys.platform == 'linux2': device = torch.device("cuda")
if sys.platform == 'win32': device = torch.device("cpu")

# -----------------------------------------------------------------------------------
input_normalization = False
num_sample_used_test = 1000
if device.type == 'cpu':  batch_size, num_workers = 1, 1
if device.type == 'cuda': batch_size, num_workers = 1, 1

# -----------------------------------------------------------------------------------
# Choose data set
# data_set_name = 'tinymicro_synth'
# data_set_name = 'tinymicro_real'
# data_set_name = 'lung3_synth'
data_set_name = 'biosr_real'
# data_set_name = 'msi_synth'

# -----------------------------------------------------------------------------------
if input_normalization == True:
    mean_normlize = np.array([0.4488, 0.4371, 0.4040])
    std_normlize  = np.array([1.0, 1.0, 1.0])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_normlize, std=std_normlize, inplace=True),
        ])
    data_transform_back = transforms.Compose([
        transforms.Normalize(mean= - mean_normlize / std_normlize, std=1.0 / std_normlize),
        ])
if input_normalization == False:
    data_transform = transforms.ToTensor()
    data_transform_back = None

# -----------------------------------------------------------------------------------
# TinyMicro (synth)
if data_set_name == 'tinymicro_synth':
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')
    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization, scale_factor = (False, False), 4 # (LR, HR)
    in_channels, data_range, init_mode = 3, 255, 'bicubic'

# -----------------------------------------------------------------------------------
# TinyMicro (real)
if data_set_name == 'tinymicro_real':
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data')
    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization, scale_factor = (False, False), 4
    in_channels, data_range, init_mode = 3, 255, 'net'
    # in_channels, data_range, init_mode = 3, 255, 'pre-trained'

# -----------------------------------------------------------------------------------
# BioSR
if data_set_name == 'biosr_real':
    name_specimen = 'F-actin_Nonlinear'
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'

    hr_root_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'test', 'GT')
    lr_root_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'test', 'WF')
    
    hr_txt_file_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'test.txt') 
    lr_txt_file_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'test.txt') 
    normalization, scale_factor = (False, False), 3
    in_channels, data_range, init_mode = 1, 1.0, 'ave_bicubic'

# -----------------------------------------------------------------------------------
# Lung3 (synth) 
if data_set_name == 'lung3_synth':
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'

    hr_root_path = os.path.join(root_path, 'Lung3', 'data_transform')
    lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'test', 'sf_4_k_2.0_n_gaussian_std_0.03_bin_ave')
    hr_txt_file_path = os.path.join(root_path, 'Lung3', 'test.txt') 
    lr_txt_file_path = os.path.join(root_path, 'Lung3', 'test.txt') 
    normalization, scale_factor = (False, False), 4

    in_channels, data_range, init_mode = 1, 1.0, 'ave_bicubic'
# -----------------------------------------------------------------------------------
# MSI
if data_set_name == 'msi_synth': pass
# -----------------------------------------------------------------------------------

test_data = dataset_utils.SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
        hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
        transform=data_transform, normalization=normalization, id_range=[0, num_sample_used_test])

test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers)

print('Model construction ...',end=' ')

# -----------------------------------------------------------------------------------
# SRCNN
srcnn_dict = {}
srcnn_dict['name']  = 'SRCNN'
srcnn_dict['model'] = srcnn.SRCNN(in_channels=in_channels, out_channels=in_channels, scale_factor=scale_factor).to(device)
# -----------------------------------------------------------------------------------
# SRResNet
srresnet_dict = {}
srresnet_dict['name']  = 'SRResNet'
srresnet_dict['model'] = srresnet.SRResNet(num_channels=in_channels, scale_factor=scale_factor).to(device)
# -----------------------------------------------------------------------------------
# RRDBNet
n_blocks, n_features = 23, 64
rrdbnet_dict = {}
rrdbnet_dict['name'] = 'RRDBNet'
rrdbnet_dict['model'] = rrdbnet.RRDBNet(scale_factor=scale_factor, in_channels=in_channels,\
    out_channels=in_channels, n_features=n_features, n_blocks=n_blocks, growth_channels=n_features//2, bias=True).to(device)
# -----------------------------------------------------------------------------------
# EDSR
edsr_dict = {}
edsr_dict['name']  = 'EDSR'
edsr_dict['model'] = edsr.EDSR(scale=scale_factor, n_colors=in_channels, n_resblocks=16, n_features=128,\
    kernel_size=3, res_scale=0.1).to(device)
# -----------------------------------------------------------------------------------
# RCAN
rcan_dict = {}
rcan_dict['name']  = 'RCAN'
rcan_dict['model'] = rcan.RCAN(scale=scale_factor, n_colors=in_channels, n_resgroups=5, n_resblocks=10,\
    n_features=64, kernel_size=3, reduction=16, res_scale=1.0).to(device)
# -----------------------------------------------------------------------------------
# SRRLN
rln_dict = {}
rln_dict['name']  = 'RLN'
rln_dict['model'] = rln.RLN(scale=scale_factor, in_channels=in_channels, n_features=4, kernel_size=3).to(device)
# -----------------------------------------------------------------------------------
# SwinIR
swinir_dict = {}
swinir_dict['name'] = 'SwinIR'
swinir_dict['model'] = swinir.SwinIR_cus(upscale=scale_factor, img_size=(128, 128), window_size=8,\
    in_chans=in_channels, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,\
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle').to(device)
# -----------------------------------------------------------------------------------
# DFCAN
dfcan_dict = {}
dfcan_dict['name'] = 'DFCAN'
dfcan_dict['model'] = dfcan.DFCAN(in_channels=in_channels, scale_factor=scale_factor, num_features=64, num_groups=4).to(device)
# -----------------------------------------------------------------------------------
# RLSR-on, only the network part
# -----------------------------------------------------------------------------------
initializer, forw_proj = None, None
if data_set_name == 'lung3_synth': num_block, n_features, backbone_type, RL_version, use_prior, prior_type = (1, 5), (2,8), 'rrdb', 'ISRA', True, 'learned'
if data_set_name == 'biosr_real':  num_block, n_features, backbone_type, RL_version, use_prior, prior_type = (1, 5), (8,8), 'rrdb', 'ISRA', True, 'learned'
rlsr_on_dict = {}
rlsr_on_dict['name']  = 'RLSR-on'
rlsr_on_dict['model'] = rlsr.RLSR(img_size=(128, 128), scale=scale_factor, in_channels=in_channels,\
    n_features=n_features, n_blocks=num_block, n_iter=1, bias=True, \
    init_mode=init_mode, backbone_type=backbone_type, window_size=8,\
    mlp_ratio=2, kernel_size=3, only_net=True, upsample_mode='conv_up', RL_version=RL_version,\
    constraint_01=True, bn=True, cat_x=False, use_prior=use_prior, forw_proj=forw_proj,\
    train_only_prior=False, fpm=1, bp_ks=3, prior_type=prior_type, prior_inchannels=1, bpm=1).to(device)
# -----------------------------------------------------------------------------------
# Initializer
# -----------------------------------------------------------------------------------
if init_mode == 'pre-trained':
    
    init_net_dict = {}
    init_net_dict['name'] = 'Initializer'
    init_net_dict['model'] = rlsr.Initializer(in_channels=in_channels, scale=4, kernel_size=3).to(device)
    init_net_dict['epoch'] = '95000'
    init_net_dict['model_ver'] = 'initializer_bs_3_lr_0.001'

# -----------------------------------------------------------------------------------
# RLSR
# -----------------------------------------------------------------------------------
fp_mode = 'pre-trained' # 'known', 'pre-trained', None
initializer, forw_proj = None, None
if fp_mode == 'pre-trained':
    if data_set_name == 'lung3_synth':
        model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_16_lr_0.001_1_2_bin_ave', 'epoch_24860.pt')
        # model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_16_lr_0.001_1_4_bin_ave', 'epoch_16272.pt')
        num_blocks, num_fea = 1, 2
    if data_set_name == 'biosr_real':
        model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_4_lr_0.001_2_4_bin_ave', 'epoch_95000.pt')
        num_blocks, num_fea = 2, 4
    model_para = torch.load(model_path, map_location=device)
    backbone  = rrdbnet.backbone(in_channels=in_channels, n_features=num_fea, n_blocks=num_blocks, growth_channels=num_fea // 2, bias=True)
    forw_proj = rlsr.ForwardProject(backbone=backbone, in_channels=in_channels,\
        n_features=num_fea, scale_factor=scale_factor, bias=True, bn=False,\
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

num_iter, num_block, n_features, backbone_type, RL_version = 1, (2,5), (4,8), 'rrdb', 'ISRA' # 'RL', 'ISRA'
batchnorm, cat_x = True, False
use_prior, train_only_prior, prior_type, lambda_prior = True, False, 'learned', 1.0 #' TV'
prior_inchannels = 1

rlsr_dict = {}
rlsr_dict['name']  = 'RLSR'
rlsr_dict['model'] = rlsr.RLSR(img_size=(128, 128), scale=scale_factor, in_channels=in_channels,\
    n_features=n_features, n_blocks=num_block, n_iter=num_iter, bias=True, \
    init_mode=init_mode, backbone_type=backbone_type, window_size=8,\
    mlp_ratio=2, kernel_size=3, only_net=False, output_mode='each-iter-test',pixel_binning_mode='ave',\
    initializer=initializer, upsample_mode='conv_up', use_prior=use_prior, RL_version=RL_version,\
    constraint_01=True, forw_proj=forw_proj, bn=batchnorm, cat_x=cat_x,\
    train_only_prior=train_only_prior, fpm=1, bp_ks=3, prior_type=prior_type, lambda_prior=lambda_prior,
    prior_inchannels=prior_inchannels, bpm=1).to(device)

if data_set_name == 'tinymicro_synth':
    rlsr_dict['model_ver'], rlsr_dict['epoch'] = 'rlsr_bs_2_lr_0.001_iter_2_block_(5,5)_feature_16_mul_ig_1.0_rrdb_bicubic_ay_alterupdate_direct_sigmoid', '95000'

if data_set_name == 'tinymicro_real':
    if init_mode == 'net':
        rlsr_dict['model_ver'], rlsr_dict['epoch'] = 'rlsr_bs_3_lr_0.001_iter_2_block_(5,5)_feature_16_mul_ig_1.0_rrdb_net_ay_alterupdate_direct_sigmoid', '95000'
    if init_mode == 'pre-trained':
        rlsr_dict['model_ver'], rlsr_dict['epoch'] = 'rlsr_bs_3_lr_0.001_iter_2_block_(5,5)_feature_16_mul_ig_1.0_rrdb_pre-trained_ay_alterupdate_direct_sigmoid_pretrained', '95000'

if data_set_name == 'lung3_synth':
    srcnn_dict['model_ver'], srcnn_dict['epoch']        = 'srcnn_bs_4_lr_0.0005', '95000'
    srresnet_dict['model_ver'], srresnet_dict['epoch']  = 'srresnet_bs_4_lr_0.0005', '95000'
    rrdbnet_dict['model_ver'], rrdbnet_dict['epoch']    = 'rrdbnet_bs_3_lr_0.0001_23_64', '95000'
    edsr_dict['model_ver'], edsr_dict['epoch']          = 'edsr_bs_4_lr_0.0005', '95000'
    rcan_dict['model_ver'], rcan_dict['epoch']          = 'rcan_bs_4_lr_0.0001', '95000'
    rln_dict['model_ver'], rln_dict['epoch']            = 'rln_bs_4_lr_0.001', '95000'
    swinir_dict['model_ver'], swinir_dict['epoch']      = 'swinir_bs_3_lr_0.0001', '95000'
    dfcan_dict['model_ver'], dfcan_dict['epoch']        = 'dfcan_bs_4_lr_0.0001', '95000'
    rlsr_on_dict['model_ver'], rlsr_on_dict['epoch']    = 'rlsr_bs_4_lr_0.001_block_(1,5)_feature_8_rrdb_only_net', '95000'
    rlsr_dict['model_ver'], rlsr_dict['epoch']          = 'rlsr_bs_16_lr_0.01_iter_1_block_(1,5)_fea_8_bb_rrdb_init_ave_bicubic_model_ISRA_bin_ave_fp_pre-trained_fpm_1_bn_pri_learned_1_bpm_1', '70000'

if data_set_name == 'biosr_real':
    srcnn_dict['model_ver'], srcnn_dict['epoch']        = 'srcnn_bs_4_lr_0.0001',   '95000'
    srresnet_dict['model_ver'], srresnet_dict['epoch']  = 'srresnet_bs_4_lr_0.001', '95000'
    rrdbnet_dict['model_ver'], rrdbnet_dict['epoch']    = 'rrdbnet_bs_3_lr_0.0001_23_64', '95000'
    edsr_dict['model_ver'], edsr_dict['epoch']          = 'edsr_bs_4_lr_0.0001',    '95000'
    rcan_dict['model_ver'], rcan_dict['epoch']          = 'rcan_bs_4_lr_0.0001',    '95000'
    rln_dict['model_ver'], rln_dict['epoch']            = 'rln_bs_4_lr_0.001',      '95000'
    swinir_dict['model_ver'], swinir_dict['epoch']      = 'swinir_bs_3_lr_0.0001',  '95000'
    dfcan_dict['model_ver'], dfcan_dict['epoch']        = 'dfcan_bs_4_lr_0.0001',   '95000'
    rlsr_on_dict['model_ver'], rlsr_on_dict['epoch']    = 'rlsr_bs_4_lr_0.001_block_(1,5)_feature_8_rrdb_only_net_pri_learned', '95000'
    rlsr_dict['model_ver'], rlsr_dict['epoch']          = 'rlsr_bs_4_lr_0.001_iter_1_block_(1,5)_fea_8_bb_rrdb_init_ave_bicubic_model_ISRA_bin_ave_fp_pre-trained_fpm_1_bn_pri_learned_1_bpm_1_24', '95000'
    # rlsr_dict['model_ver'], rlsr_dict['epoch']          = 'rlsr_bs_16_lr_0.005_iter_1_block_(1,5)_fea_8_bb_rrdb_init_ave_bicubic_model_RL_bin_ave_fp_pre-trained_fpm_1_bn_pri_learned_1_bpm_1_24', '75000'

print('(done)')
# -----------------------------------------------------------------------------------
kernel = torch.Tensor(ip.gaussian_kernel(n=25, std=2.0)).to(device)
RL_deconv = rld.RLD(kernel=kernel, data_range=(0, 1.0), pbar_disable=True, device=device)

# -----------------------------------------------------------------------------------
pure_net = [srcnn_dict, srresnet_dict, edsr_dict, rcan_dict, rln_dict, swinir_dict, dfcan_dict, rrdbnet_dict, rlsr_on_dict]
if init_mode == 'pre-trained': pure_net = pure_net.append(init_net_dict)
conventional_methods = ['nearest', 'bicubic', 'Bicubic+RLD']
model_based_net = [rlsr_dict]
deep_learning_methods = pure_net + model_based_net

methods_eva = ['nearest', 'bicubic', 'Bicubic+RLD', srcnn_dict, edsr_dict, rcan_dict, srresnet_dict, rrdbnet_dict, swinir_dict, dfcan_dict, rln_dict, rlsr_on_dict, rlsr_dict]
# methods_eva = [init_net_dict, rlsr_on_dict, rlsr_dict]
# methods_eva = [rlsr_dict, rlsr_on_dict, dfcan_dict]
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Counting parameters
for dicts in deep_learning_methods:
    print('Model: {:<10s}, '.format(dicts['name']), end='')
    eva.count_parameters(dicts['model'])

# ###################################################################################
# Evaluation on single image
# ###################################################################################
fig_dir = os.path.join('outputs', 'figures', data_set_name)
if os.path.exists(fig_dir) == False: os.makedirs(fig_dir, exist_ok=True)
# -----------------------------------------------------------------------------------
if in_channels == 3: t2i = dataset_utils.tensor2rgb
if in_channels == 1: t2i = dataset_utils.tensor2gray
dtb = data_transform_back
# -----------------------------------------------------------------------------------
def predict(model_dict, data): 
    model_path = os.path.join('checkpoints', data_set_name, model_dict['model_ver'], 'epoch_{}.pt'.format(model_dict['epoch']))
    print('Model: ', model_path, end=' ')
    model_para = torch.load(model_path, map_location=device)
    model = model_dict['model']
    model.load_state_dict(model_para['model_state_dict'], strict=False)
    model.eval()
    pred = model(data)
    if input_normalization == True: dtb(pred)
    pred = t2i(pred)
    print('(done)')
    return pred
# -----------------------------------------------------------------------------------
if data_set_name == 'lung3_synth':  id_data = 989
if data_set_name == 'biosr_real':   id_data = 100

ds = test_data[id_data]
x, y = torch.unsqueeze(ds['lr'], 0).to(device), torch.unsqueeze(ds['hr'], 0).to(device)
print('Sample size: ', x.shape)
if input_normalization == True: x, y = dtb(x), dtb(y)
y = t2i(y)
# -----------------------------------------------------------------------------------
# convolutional methods
x_nearest = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='nearest')
x_bicubic = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bicubic')
y_rld = RL_deconv.decov(x_bicubic, 100) # [W, H, C] or [W. H]

x_nearest, x_bicubic, y_rld = t2i(x_nearest), t2i(x_bicubic), t2i(y_rld)
# -----------------------------------------------------------------------------------
# deep learning methods
if init_mode == 'pre-trained': y_init = t2i(predict(init_net_dict, x))
y_rlsr_on   = predict(rlsr_on_dict, x)
y_rlsr      = predict(rlsr_dict, x)[-1]
y_srcnn     = predict(srcnn_dict, x)
y_srresnet  = predict(srresnet_dict, x)
y_rrdbnet   = predict(rrdbnet_dict, x)
y_edsr      = predict(edsr_dict, x)
y_rcan      = predict(rcan_dict, x)
y_srrln     = predict(rln_dict, x)
y_swinir    = predict(swinir_dict, x)
y_dfcan     = predict(dfcan_dict, x)
# ------------------------------------------------------------------------------------
# # post-processing
# if data_set_name == 'biosr_real':
#     y = dataset_utils.percentile_norm(y, p_low=0.1, p_high=99.9)
#     x_nearest   = dataset_utils.linear_transform(x_nearest, y)
#     x_bicubic   = dataset_utils.linear_transform(x_bicubic, y)
#     y_rld       = dataset_utils.linear_transform(y_rld, y)
#     y_rlsr_on   = dataset_utils.linear_transform(y_rlsr_on, y)
#     y_rlsr      = dataset_utils.linear_transform(y_rlsr, y)
#     y_srcnn     = dataset_utils.linear_transform(y_srcnn, y)
#     y_srresnet  = dataset_utils.linear_transform(y_srresnet, y)
#     y_rrdbnet   = dataset_utils.linear_transform(y_rrdbnet, y)
#     y_edsr      = dataset_utils.linear_transform(y_edsr, y)
#     y_rcan      = dataset_utils.linear_transform(y_rcan, y)
#     y_srrln     = dataset_utils.linear_transform(y_srrln, y)
#     y_swinir    = dataset_utils.linear_transform(y_swinir, y)
#     y_dfcan     = dataset_utils.linear_transform(y_dfcan, y)

# ------------------------------------------------------------------------------------
nr, nc = 2, 7
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=600, figsize=(2.4 * nc, 2.4 * nr), constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
if data_set_name == 'biosr_real':   vmax_tmp = np.max(y)
if data_set_name == 'lung3_synth':  vmax_tmp = 1.0
def show_res(ax, img, gt, title):
    if in_channels == 3: ax.imshow(img[0])
    if in_channels == 1: ax.imshow(img[0, ..., 0], cmap='gray', vmin=0.0, vmax=vmax_tmp)
    # ax.set_title(title + ' ({:>.2f}|{:>.4f})'.format(eva.PSNR(gt[0], img[0], data_range=data_range), eva.SSIM(gt[0], img[0], data_range=data_range)))
    # ax.set_title(title + ' ({:>.4f}|{:>.4f})'.format(eva.NRMSE(gt[0], img[0]), eva.SSIM(gt[0], img[0], data_range=data_range)))
    ax.set_title(title + ' ({:>.4f}|{:>.4f})'.format(eva.NRMSE(gt[0], img[0]), eva.MSSSIM(gt[0], img[0], data_range=data_range)[0]))

# ------------------------------------------------------------------------------------
show_res(axes[0,0], x_nearest,  y, 'LR')
show_res(axes[0,1], x_bicubic,  y, 'Bicubic')
show_res(axes[0,2], y_rld,      y, 'Bicubic+RLD')
show_res(axes[0,3], y_srcnn,    y, srcnn_dict['name'])
show_res(axes[0,4], y_edsr,     y, edsr_dict['name'])
show_res(axes[0,5], y_rcan,     y, rcan_dict['name'])
show_res(axes[0,6], y_srresnet, y, srresnet_dict['name'])
if init_mode == 'pre-trained': show_res(axes[0,6], y_init, y, init_net_dict['name'])

if in_channels == 3: axes[1,0].imshow(y[0])
if in_channels == 1: axes[1,0].imshow(y[0, ..., 0], cmap='gray', vmin=0.0, vmax=vmax_tmp)
axes[1,0].set_title('HR')
show_res(axes[1,1], y_swinir,   y, swinir_dict['name'])
show_res(axes[1,2], y_dfcan,    y, dfcan_dict['name'])
show_res(axes[1,4], y_srrln,    y, rln_dict['name'])
show_res(axes[1,5], y_rlsr_on,  y, rlsr_on_dict['name'])
show_res(axes[1,6], y_rlsr,     y, rlsr_dict['name'])

plt.savefig(os.path.join(fig_dir, 'comparison_' + str(id_data)))

# os._exit(0)

num_sample = len(test_data)
print('='*98)
print('Evaluation on testing dataset ....')
print('Total number of sample: {}'.format(num_sample))

def post_process(y_pred, y):
    y_norm = dataset_utils.percentile_norm(y, p_low=0.1, p_high=99.9)
    y_pred = dataset_utils.linear_transform(y_pred, y_norm)
    return y_pred

# ###################################################################################
# Evaluation on test dataset
# ###################################################################################
for method in methods_eva:
    # -------------------------------------------------------------------------------
    print('-'*98)
    ssim, psnr, nrmse, msssim = [], [], [], []
    start_time = time.time()
    # -------------------------------------------------------------------------------
    # convolutional methods
    if method in conventional_methods:
        print('Evaluate using ', method)
        pbar = tqdm.tqdm(total=num_sample, desc=method, leave=True, ncols=100, disable=(device.type == 'cuda'))
        # ---------------------------------------------------------------------------
        for i_sample in range(num_sample):
            x = torch.unsqueeze(test_data[i_sample]['lr'], 0).to(device)
            y = torch.unsqueeze(test_data[i_sample]['hr'], 0).to(device)
            
            if method in ['nearest', 'bicubic']:
                y_pred = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=method)

            if method in ['Bicubic+RLD']:
                x_bicubic = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bicubic')
                y_pred = RL_deconv.decov(x_bicubic, 100) # [W, H, C] or [W. H]

            y, y_pred = t2i(y), t2i(y_pred)

            # y_pred = post_process(y_pred, y)

            if device.type == 'cuda': 
                print('#', end='')
                if (i_sample + 1) % 100 == 0: print(str(i_sample) + '\n', end='')

            psnr.append(    eva.PSNR(   y[0], y_pred[0], data_range=data_range))
            ssim.append(    eva.SSIM(   y[0], y_pred[0], data_range=data_range))
            nrmse.append(   eva.NRMSE(  y[0], y_pred[0]))
            msssim.append(  eva.MSSSIM( y[0], y_pred[0], data_range=data_range)[0])
            pbar.update(1)
    
    # -------------------------------------------------------------------------------
    # Deep learning methods
    if method in deep_learning_methods:
        print('Evaluate using ', method['name'])
        # load model
        try:
            model_path = os.path.join('checkpoints', data_set_name, method['model_ver'], 'epoch_{}.pt'.format(method['epoch']))
            print('Model: ', model_path)
            model_para = torch.load(model_path, map_location=device)
            model = method['model']
            model.load_state_dict(model_para['model_state_dict'], strict=False)
            model.eval()
            # print(model.state_dict())
        except:
            model = None
            print('[Error] No model!')
        # ---------------------------------------------------------------------------
        pbar = tqdm.tqdm(total=num_sample, desc=method['name'], leave=True, ncols=100,\
            disable=(device.type == 'cuda'))
        # ---------------------------------------------------------------------------
        for i_batch, sample in enumerate(test_dataloader):
            x, y = sample['lr'].to(device), sample['hr'].to(device)
            y_pred = model(x)

            y_pred, y = t2i(y_pred), t2i(y)

            # -----------------------------------------------------------------------
            if method in pure_net:
                for i_sample in range(y_pred.shape[0]):
                    y_tmp, y_pred_tmp = y[i_sample], y_pred[i_sample]
                    y_pred_tmp = post_process(y_pred_tmp, y_tmp)
                    psnr.append(    eva.PSNR(   y_tmp, y_pred_tmp, data_range=data_range))
                    ssim.append(    eva.SSIM(   y_tmp, y_pred_tmp, data_range=data_range))
                    nrmse.append(   eva.NRMSE(  y_tmp, y_pred_tmp))
                    msssim.append(  eva.MSSSIM( y_tmp, y_pred_tmp, data_range=data_range)[0])
            # -----------------------------------------------------------------------
            if method in model_based_net:
                psnr_batch, ssim_batch, nrmse_batch, msssim_batch = [], [], [], []
                for i_iter in range(y_pred.shape[0]):
                    psnr_iter, ssim_iter, nrmse_iter, msssim_iter = [], [], [], []
                    for i_sample in range(y_pred.shape[1]):
                        y_tmp, y_pred_tmp = y[i_sample], y_pred[i_iter, i_sample]
                        # y_pred_tmp = post_process(y_pred_tmp, y_tmp)
                        psnr_iter.append(   eva.PSNR(   y_tmp, y_pred_tmp, data_range=data_range))
                        ssim_iter.append(   eva.SSIM(   y_tmp, y_pred_tmp, data_range=data_range))
                        nrmse_iter.append(  eva.NRMSE(  y_tmp, y_pred_tmp))
                        msssim_iter.append( eva.MSSSIM( y_tmp, y_pred_tmp, data_range=data_range)[0])
                    psnr_batch.append(  psnr_iter)
                    ssim_batch.append(  ssim_iter)
                    nrmse_batch.append( nrmse_iter)
                    msssim_batch.append(msssim_iter)
                psnr.append(psnr_batch)
                ssim.append(ssim_batch)
                nrmse.append(nrmse_batch)
                msssim.append(msssim_batch)
                
            # ------------------------------------------------------------------------
            # print processing
            if device.type == 'cuda': 
                print('#', end='')
                i_sample = (i_batch + 1) * batch_size
                if i_sample % 100 == 0: print(str(i_sample) + '\n', end='')
            pbar.update(batch_size)

    # print results
    # ---------------------------------------------------------------------------
    if (method in conventional_methods) or (method in pure_net):
        ssim, psnr, nrmse, msssim = np.array(ssim), np.array(psnr), np.array(nrmse), np.array(msssim)
        print('Evaluation Metrics : SSIM: {:>.4f}({:>.4f}), PSNR: {:>.4f}({:>.4f}), NRMSE: {:>.4f}({:>.4f}), MS-SSIM: {:>.4f}({:>.4f})'\
            .format(np.mean(ssim), np.std(ssim), np.mean(psnr), np.std(psnr), np.mean(nrmse), np.std(nrmse), np.mean(msssim), np.std(msssim)))
    
    # ---------------------------------------------------------------------------
    if method in model_based_net:
        ssim, psnr, nrmse, msssim = np.array(ssim), np.array(psnr), np.array(nrmse), np.array(msssim) # [batch, iter, sample]
        ssim = np.transpose(ssim, axes=(1, 0, 2))
        psnr = np.transpose(psnr, axes=(1, 0, 2))
        nrmse = np.transpose(nrmse, axes=(1, 0, 2))
        msssim = np.transpose(msssim, axes=(1, 0, 2))
        for i_iter in range(ssim.shape[0]):
            print('Evaluation Metrics : [iter {}] SSIM: {:>.4f}({:>.4f}), PSNR: {:>.4f}({:>.4f}), NRMSE: {:>.4f}({:>.4f}), MS-SSIM: {:>.4f}({:>.4f})'\
                .format(i_iter,\
                np.mean(ssim[i_iter]), np.std(ssim[i_iter]),\
                np.mean(psnr[i_iter]), np.std(psnr[i_iter]),\
                np.mean(nrmse[i_iter]), np.std(nrmse[i_iter]),\
                np.mean(msssim[i_iter]), np.std(msssim[i_iter]),\
                    ))

    end_time = time.time()
    pbar.close()
    print('Processing time: {:>.2f} s, average time: {:>.2f} s'.format(end_time - start_time,\
        (end_time - start_time) / num_sample))
