import torch, os, tqdm, time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np

from utils import dataset_utils
from utils import evaluation as eva
from models import srcnn, srresnet, edsr, rln, rcan, swinir, rlsr, rrdbnet, ddn, dfcan
import loss_functions as loss_func
import sys

###################################################################################
def rescale(tensor, scale=1.0): return torch.clamp(tensor, min=0.0 ,max=1.0) * scale

# ---------------------------------------------------------------------------------
print('='*98)
if sys.platform == 'win32': device = torch.device("cpu") # set device
if sys.platform == 'linux' or sys.platform == 'linux2': device = torch.device("cuda") 
print('Device: ', device)
if device.type == 'cuda': num_workers = 6
if device.type == 'cpu':  num_workers = 0
# ---------------------------------------------------------------------------------
torch.manual_seed(7)          
input_normalization = 0
checkpoint_path = 'checkpoints'
validation_disable = False
# ---------------------------------------------------------------------------------
# choose data set
# ---------------------------------------------------------------------------------
# data_set_name = 'tinymicro_synth'
# data_set_name = 'tinymicro_real'
# data_set_name = 'lung3_synth'
data_set_name = 'biosr_real'
# data_set_name = 'msi_synth'

# ---------------------------------------------------------------------------------
# choose model
# ---------------------------------------------------------------------------------
# model_name = 'srcnn'
# model_name = 'srresnet'
# model_name = 'rrdbnet'
# model_name = 'edsr'
# model_name = 'rln'
# model_name = 'ddn'
# model_name = 'rcan'
# model_name = 'swinir'
# model_name = 'dfcan'
# model_name = 'initializer'
# model_name = 'forw_proj'
# model_name, only_net = 'rlsr', True
model_name, only_net = 'rlsr', False
# ---------------------------------------------------------------------------------
if model_name in ['srcnn', 'srresnet', 'edsr', 'rln', 'ddn', 'rcan', 'swinir', 'initializer', 'dfcan']:
    only_net, output_mode, model_suffix = True, 'last-one', '_mse_ssim'
    lambda_loss = 0.1

if model_name == 'forw_proj':
    only_net, output_mode, n_blocks, n_features, pixel_binning_mode = True, 'last-one', 1, 2, 'ave'
    model_suffix = '_{}_{}_bin_{}_newdata'.format(n_blocks, n_features, pixel_binning_mode)
    lambda_loss = 0.0

if model_name == 'rrdbnet': 
    only_net, output_mode, n_blocks, n_features = True, 'last-one', 23, 64
    model_suffix = '_{}_{}_newdata'.format(n_blocks, n_features)
    lambda_loss = 0.0
# ---------------------------------------------------------------------------------
if model_name == 'rlsr':
    if data_set_name == 'lung3_synth': backbone_type, n_blocks, n_features, kernel_size, bp_ks = 'rrdb', (1,5), (4,8), 3, 3
    if data_set_name == 'biosr_real':  backbone_type, n_blocks, n_features, kernel_size, bp_ks = 'rrdb', (2,4), (4,8), 3, 3
    upsample_mode, RL_version, pixel_binning_mode = 'conv_up', 'RL', 'ave' # 'conv_up', 'up_conv', 'RL', 'ISRA'
    constraint_01, batchnorm, cat_x = True, True, False
    if only_net == False: fp_mode, fpm, bpm = 'pre-trained', 1, 1 # 'known', 'pre-trained', None
    if only_net == True:  fp_mode, fpm, bpm = None, 1, 1 # 'known', 'pre-trained', None
    use_prior, train_only_prior, prior_type, lambda_prior, prior_inchannels, prior_bn = True, False, 'learned', 1.0, 1, False # 'learned', 'TV'
    if use_prior == False: prior_type = 'none'

    n_iter = 1
    lambda_loss = 0.0

    if data_set_name in ['tinymicro_synth', 'lung3_synth', 'biosr_real', 'msi_synth']: init_mode = 'ave_bicubic' # 'bicubic', 'ave_bicubic', 'nearest'
    if data_set_name in ['tinymicro_real']: init_mode = 'net' # 'net', 'pre-trained'

    if only_net == True:
        output_mode  = 'last-one'
        model_suffix = '_block_({},{})_feature_({},{})_{}_only_net_pri_{}'.format(n_blocks[0], n_blocks[1], n_features[0], n_features[1], backbone_type, prior_type)

    if only_net == False:
        output_mode  = 'each-iter-train-prior'
        # output_mode  = 'each-iter-train'
        model_suffix = '_iter_{}_block_({},{})_fea_({},{})_bb_{}_init_{}_model_{}_bin_{}_fp_{}_fpm_{}_bn_pri_{}_{}_bpm_{}_lam_{}_24'\
            .format(n_iter, n_blocks[0], n_blocks[1], n_features[0], n_features[1], backbone_type, init_mode, RL_version, pixel_binning_mode, fp_mode, fpm,\
            prior_type, prior_inchannels, bpm, lambda_loss)
# ---------------------------------------------------------------------------------
batch_size = 4
start_learning_rate = 0.001
if data_set_name in ['tinymicro_synth', 'tinymicro_real']: epochs = 3
if data_set_name in ['lung3_synth']:    epochs = 62 # 36, 40, 55
if data_set_name in ['biosr_real']:     epochs = 24 # 15, 21
# ---------------------------------------------------------------------------------
save_every_iter, plot_every_iter, val_every_iter = 5000, 50, 5000
warm_up = 5000
# learning rate scheduler
use_lr_schedule = True
scheduler_cus = {}
scheduler_cus['lr']    = start_learning_rate
scheduler_cus['every'] = 10000
scheduler_cus['rate']  = 0.5
scheduler_cus['min']   = 0.0000001

if use_lr_schedule == True:
    print('Use learning rate decay technique.')
    print('Start learning rate: {}, decay every: {}, decay rate: {}'.format(\
        scheduler_cus['lr'], scheduler_cus['every'], scheduler_cus['rate']) )
if use_lr_schedule == False:
    print('Constant learning rate: ', start_learning_rate)

# #################################################################################
# Data 
# #################################################################################
# input normalization
if input_normalization == True:
    print('Input Normalization (on)')
    mean_normlize = np.array([0.4488, 0.4371, 0.4040])
    std_normlize  = np.array([1.0, 1.0, 1.0])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_normlize,std=std_normlize,inplace=True),
        ])
    data_transform_back = transforms.Compose([
        transforms.Normalize(mean=-mean_normlize/std_normlize,std=1.0/std_normlize),
        ])

if input_normalization == False:
    print('Input Normalization (off)')
    data_transform = transforms.ToTensor()
    data_transform_back = None

# ---------------------------------------------------------------------------------
# Training data
if data_set_name == 'tinymicro_synth': # TinyMicro (synth)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'train', 'sf_4_k_2.0_gaussian_mix_ave')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'lr.txt')
    normalization, in_channels, scale_factor = (False, False), 3, 4

if data_set_name == 'tinymicro_real': # TinyMicro (real)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'train_txt', 'lr.txt')
    normalization, in_channels, scale_factor = (False, False), 3, 4

if data_set_name == 'biosr_real':
    name_specimen = 'F-actin_Nonlinear'
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'
    hr_root_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'train', 'GT')
    lr_root_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'train', 'WF')

    hr_txt_file_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'train.txt') 
    lr_txt_file_path = os.path.join(root_path, 'BioSR', 'data_transform', name_specimen, 'train.txt') 
    normalization, in_channels, scale_factor = (False, False), 1, 3
    training_size = 18663

if data_set_name == 'lung3_synth': # Lung3 (synth)
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'
    hr_root_path = os.path.join(root_path, 'Lung3', 'data_transform')
    lr_root_path = os.path.join(root_path, 'Lung3', 'data_synth', 'train', 'sf_4_k_2.0_n_gaussian_std_mix_bin_ave')

    hr_txt_file_path = os.path.join(root_path, 'Lung3', 'train.txt') 
    lr_txt_file_path = os.path.join(root_path, 'Lung3', 'train.txt') 
    normalization, in_channels, scale_factor = (False, False), 1, 4
    training_size = 6503

if data_set_name == 'msi_synth': pass
# ---------------------------------------------------------------------------------
# Training data
if validation_disable == True: training_size = -1
training_data = dataset_utils.SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
        hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
        transform=data_transform, normalization=normalization, id_range=[0, training_size])

train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# ---------------------------------------------------------------------------------
# Validation data
if validation_disable == False:
    validation_data = dataset_utils.SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
            hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
            transform=data_transform, normalization=normalization, id_range=[training_size, -1])

    valid_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# #################################################################################
# Model
# #################################################################################
if model_name == 'srcnn': 
    model = srcnn.SRCNN(in_channels=in_channels, out_channels=in_channels, scale_factor=scale_factor).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'srresnet': 
    model = srresnet.SRResNet(num_channels=in_channels, scale_factor=scale_factor).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'rrdbnet': 
    # original: n_features, n_blocks, growth_channels = 64, 23, 32
    model = rrdbnet.RRDBNet(scale_factor=scale_factor, in_channels=in_channels,\
        out_channels=in_channels, n_features=n_features, n_blocks=n_blocks,\
        growth_channels=n_features//2, bias=True).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'edsr':
    model = edsr.EDSR(scale=scale_factor, n_colors=in_channels, n_resblocks=16, n_features=128,\
        kernel_size=3, res_scale=0.1).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'rln':
    model = rln.RLN(scale=scale_factor, in_channels=in_channels, n_features=4, kernel_size=3).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'ddn':
    model = ddn.DenseDeconNet(in_channels=in_channels, scale_factor=scale_factor).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'rcan':
    model = rcan.RCAN(scale=scale_factor, n_colors=in_channels, n_resgroups=5, n_resblocks=10,\
        n_features=64, kernel_size=3, reduction=16, res_scale=1.0).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'swinir':
    model = swinir.SwinIR_cus(upscale=scale_factor, img_size=(128, 128), window_size=8, img_range=1.,\
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,\
        upsampler='pixelshuffle', in_chans=in_channels).to(device)

if model_name == 'dfcan':
    model = dfcan.DFCAN(in_channels=in_channels, scale_factor=scale_factor, num_features=64, num_groups=4).to(device)
# ---------------------------------------------------------------------------------
if model_name == 'rlsr':
    initializer, forw_proj = None,  None
    if fp_mode == 'pre-trained':
        if data_set_name == 'lung3_synth':
            # model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_16_lr_0.001_1_2_bin_ave', 'epoch_24860.pt')
            model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_16_lr_0.001_1_4_bin_ave', 'epoch_16272.pt')
        if data_set_name == 'biosr_real':
            model_path = os.path.join('checkpoints', data_set_name, 'forw_proj_bs_4_lr_0.001_2_4_bin_ave', 'epoch_95000.pt')
        model_para = torch.load(model_path, map_location=device)
        backbone  = rrdbnet.backbone(in_channels=in_channels, n_features=n_features[0], n_blocks=n_blocks[0], growth_channels=n_features[0] // 2, bias=True)
        forw_proj = rlsr.ForwardProject(backbone=backbone, in_channels=in_channels,\
            n_features=n_features[0], scale_factor=scale_factor, bias=True, bn=False,\
            kernel_size=3, only_net=False, pixel_binning_mode='ave').to(device)
        forw_proj.load_state_dict(model_para['model_state_dict'])
        forw_proj.eval()
    
    if fp_mode == 'known':
        ks, sig = 25, 2.0
        ker = rln.gauss_kernel_2d(shape=(ks, ks), sigma=sig)
        ker = ker.repeat(repeats=(in_channels, 1, 1, 1)).to(device=device)
        padd = lambda x: torch.nn.functional.pad(input=x, pad=(ks//2, ks//2, ks//2, ks//2), mode='reflect')
        conv = lambda x: torch.nn.functional.conv2d(input=padd(x), weight=ker, stride=1, groups=in_channels)
        forw_proj = lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor)
    
    model = rlsr.RLSR(img_size=(128, 128), scale=scale_factor, in_channels=in_channels,\
        n_features=n_features, n_blocks=n_blocks, n_iter=n_iter, bias=True,\
        init_mode=init_mode, backbone_type=backbone_type, window_size=8,\
        mlp_ratio=2, kernel_size=kernel_size, only_net=only_net, output_mode=output_mode,\
        initializer=initializer, upsample_mode=upsample_mode, use_prior=use_prior,\
        RL_version=RL_version, pixel_binning_mode=pixel_binning_mode,\
        constraint_01=constraint_01, forw_proj=forw_proj, bn=batchnorm, cat_x=cat_x,\
        fpm=fpm, bp_ks=bp_ks, bpm=bpm,\
        prior_type=prior_type, train_only_prior=train_only_prior, lambda_prior=lambda_prior, prior_inchannels=prior_inchannels, prior_bn=prior_bn,\
        ).to(device)
    
    if train_only_prior == True:
        print('Only train the prior network.')
        print('Load FP and BP parameters ...')
        model_path = os.path.join('checkpoints', data_set_name,\
            # 'rlsr_bs_4_lr_0.005_iter_1_block_(1,5)_fea_8_bb_rrdb_init_ave_bicubic_model_RL_bin_ave_fp_pre-trained_fpm_1_bn_pri_none_1_bpm_1',\
            'rlsr_bs_4_lr_0.010_iter_1_block_(1,5)_fea_8_bb_rrdb_init_ave_bicubic_model_ISRA_bin_ave_fp_pre-trained_fpm_1_bn_pri_none_1_bpm_1',\
            'epoch_95000.pt')
        print('Path: ', model_path)
        model_para = torch.load(model_path, map_location=device)
        model.load_state_dict(model_para['model_state_dict'], strict=False)

# -------------------------------------------------------------------------------------
if model_name == 'initializer':
    model = rlsr.Initializer(in_channels=in_channels, scale=4, kernel_size=3).to(device)

# -------------------------------------------------------------------------------------
if model_name == 'forw_proj':
    backbone = rrdbnet.backbone(in_channels=in_channels, n_features=n_features, \
        n_blocks=n_blocks, growth_channels=n_features // 2, bias=True)
    model = rlsr.ForwardProject(backbone=backbone, in_channels=in_channels,\
        n_features=n_features, scale_factor=scale_factor, bias=True, bn=False,\
        kernel_size=3, only_net=False, pixel_binning_mode=pixel_binning_mode).to(device)
# -------------------------------------------------------------------------------------
# count parameter number
eva.count_parameters(model)
# eva.count_parameters(model.FP)
# eva.count_parameters(model.BP)
# if use_prior == True: eva.count_parameters(model.Prior)
# print(model)
# -------------------------------------------------------------------------------------
# save graph
model_path = os.path.join(checkpoint_path, data_set_name, '{}_bs_{}_lr_{}{}'.format(model_name, batch_size, start_learning_rate, model_suffix))
writer = SummaryWriter(os.path.join(model_path, 'log')) # TensorBoard writer
print('Save model to ' + model_path)

# save graph to tensorboard writer
# sample_one = next(iter(train_dataloader))
# writer.add_graph(model, sample_one['lr'].to(device))

#######################################################################################
# optimization
#######################################################################################
# loss_main = lambda x, y: torch.sum(torch.mean(torch.square(x - y), dim=(-1, -2, -3, -4)))
loss_main = lambda x, y: torch.sum(torch.mean(torch.abs(x - y), dim=(-1, -2, -3, -4)))
# loss_main = torch.nn.SmoothL1Loss(beta=0.5)
# loss_main = lambda x, y: torch.sum(torch.mean(torch.abs(x - y) / (y + 0.01), dim=(-1, -2, -3, -4)))
# loss_main = torch.nn.L1Loss()

# -------------------------------------------------------------------------------------
# loss_aux = lambda x, y: loss_func.SSIM_neg_ln(x, y)
loss_aux = lambda x, y: loss_func.SSIM_one_sub(x, y)
# weights_path = os.path.join('checkpoints', 'pretrained_model', 'vgg16-397923af.pth')
# perc_loss = loss_func.VGGPerceptualLoss(weights_path=weights_path, device=device)
# loss_aux = lambda x, y: perc_loss(x, y)
# edge_tv = rlsr.TV_grad(epsilon=0.001)
# loss_aux = lambda x, y: loss_func.mae_edge(x, y, edge_func=edge_tv)

# --------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=start_learning_rate)

# --------------------------------------------------------------------------------------
print('Start training ... ')
print(time.asctime(time.localtime(time.time())))

num_batches = len(train_dataloader)
num_batches_val = len(valid_dataloader)
print('Number of training batches: {}, validation batches: {}'.format(num_batches, num_batches_val))

for i_epoch in range(epochs):
    print('-'*98)
    print('Epoch: {}/{} | Batch size: {} | Num of Batches: {}'.format(i_epoch + 1, epochs, batch_size, num_batches))
    print('-'*98)
    # ----------------------------------------------------------------------------------
    ave_ssim, ave_psnr = 0, 0
    print_loss, print_ssim, print_psnr, print_loss_aux = [], [], [], []
    
    start_time = time.time()
    # ----------------------------------------------------------------------------------
    model.train()
    for i_batch, sample in enumerate(train_dataloader):
        i_iter = i_batch + i_epoch * num_batches # index of iteration
        x, y = sample['lr'].to(device), sample['hr'].to(device)
        optimizer.zero_grad()
        # ------------------------------------------------------------------------------
        # predict and calculate loss
        if only_net == True:
            if model_name == 'forw_proj':
                inpt, gt = y, x
                pred = model(inpt)
                loss = loss_main(pred, gt)
            else:
                inpt, gt = x, y
                pred = model(inpt)
                loss2 = loss_main(pred, gt)
                loss3 = loss_aux(pred, gt)
                loss = loss2 + lambda_loss * loss3
            # back-propagation and update
            loss.backward()
            optimizer.step()
        # ------------------------------------------------------------------------------
        if only_net == False:
            inpt, gt = x, y
            # --------------------------------------------------------------------------
            loss1 = 0
            if fp_mode not in ['pre-trained', 'known']:
                ay = model.forward_project(gt)
                loss1 = loss_main(ay, inpt) # ay
                loss1.backward()
                optimizer.step()
            # --------------------------------------------------------------------------
            pred = model(inpt)
            loss2 = loss_main(pred, gt)
            loss3 = loss_aux(pred, gt)
            loss = loss2 + lambda_loss * loss3
            loss.backward()
            optimizer.step()
        # ------------------------------------------------------------------------------
        # custom learning rate scheduler
        if use_lr_schedule == True:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus['lr']
                for g in optimizer.param_groups: g['lr'] = lr # set learning rate
            if i_iter >= warm_up:
                if (i_iter + 1 - warm_up) % scheduler_cus['every'] == 0:
                    lr = scheduler_cus['lr'] * (scheduler_cus['rate']**((i_iter + 1 - warm_up) // scheduler_cus['every']))
                    lr = np.maximum(lr, scheduler_cus['min'])
                    for g in optimizer.param_groups: g['lr'] = lr # set learning rate

        if use_lr_schedule == False:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus['lr']
                for g in optimizer.param_groups: g['lr'] = lr # set learning rate
            if i_iter >= warm_up:
                for g in optimizer.param_groups: g['lr'] = scheduler_cus['lr'] # set learning rate

        # ------------------------------------------------------------------------------
        # calculate metrics
        if data_transform_back is not None: 
            pred, gt, inpt = data_transform_back(pred), data_transform_back(gt), data_transform_back(inpt)

        if output_mode in ['each-iter-train', 'each-iter-train-prior']: out = pred[-1]
        if output_mode == 'last-one': out = pred
        # ------------------------------------------------------------------------------
        # plot loss and metrics
        if i_iter % plot_every_iter == 0:
            ave_ssim, ave_psnr = eva.measure(img_test=rescale(out, scale=1.0), img_true=rescale(gt, scale=1.0), data_range=1.0)
            if writer != None:
                writer.add_scalar('total_loss', loss, i_iter)
                writer.add_scalar('loss', loss_main(out, gt), i_iter)
                if only_net == False: writer.add_scalar('loss1', loss1, i_iter)
                if model_name == 'forw_proj': writer.add_scalar('loss1', loss, i_iter)
                writer.add_scalar('psnr', ave_psnr, i_iter)
                writer.add_scalar('ssim', ave_ssim, i_iter)
                writer.add_scalar('Leanring Rate', optimizer.param_groups[-1]['lr'], i_iter)
            if (i_iter > 5000) & (ave_psnr < 10.0):
                print('\nThe PSNR ({:>.4f}) is too low, break!'.format(ave_psnr))
                writer.flush()
                writer.close()
                os._exit(0)
        # ------------------------------------------------------------------------------
        # print and save model
        s, p = eva.measure(img_test=rescale(out, scale=1.0), img_true=rescale(gt, scale=1.0), data_range=1.0)
        print_loss.append(loss_main(out, gt).cpu().detach().numpy())
        print_loss_aux.append(loss_aux(out, gt).cpu().detach().numpy())
        print_ssim.append(s)
        print_psnr.append(p)
        print('#', end='')
        if i_iter % 200 == 0:
            print('\nEpoch: {}, Iterations: {}, loss_main: {:>.5f}, loss_aux: {:>.5f}, PSNR: {:>.5f}, SSIM: {:>.5f}'.\
                format(i_epoch, i_iter, np.mean(print_loss), np.mean(print_loss_aux), np.mean(print_psnr), np.mean(print_ssim)))
            print('Computation time: {:>.2f} s'.format(time.time() - start_time))
            start_time = time.time()
            print_loss, print_loss_aux, print_ssim, print_psnr = [], [], [], []

        # ------------------------------------------------------------------------------
        # save model and relative information
        if (i_iter + 1) % save_every_iter == 0:
            print('\nSave model ... (Epoch: {}, Iteration: {})'.format(i_epoch, i_iter))
            model_dict = {'epoch': i_epoch, 'num_iter': i_iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
            torch.save(model_dict, os.path.join(model_path, 'epoch_{}.pt'.format(i_iter + 1)))
        # ----------------------------------------------------------------------------------
        # validation
        if (i_iter % val_every_iter == 0) and (validation_disable == False):
            print('validation ...')
            running_val_loss, running_val_loss_main, running_val_ssim, running_val_psnr = 0.0, 0.0, 0.0, 0.0
            model.eval()
            for i_batch_val, sample_val in enumerate(valid_dataloader):
                x_val, y_val = sample_val['lr'].to(device), sample_val['hr'].to(device)
                pred_val  = model(x_val)
                loss2_val = loss_main(pred_val, y_val)
                loss3_val = loss_aux(pred_val, y_val)
                loss_val  = loss2_val + lambda_loss * loss3_val

                if output_mode in ['each-iter-train', 'each-iter-train-prior']: out_val = pred_val[-1]
                if output_mode == 'last-one': out_val = pred_val
                ave_ssim, ave_psnr = eva.measure(img_test=rescale(out_val, scale=1.0), img_true=rescale(y_val, scale=1.0), data_range=1.0)

                running_val_loss += loss_val.cpu().detach().numpy()
                running_val_loss_main += loss2_val.cpu().detach().numpy()
                running_val_psnr += ave_psnr
                running_val_ssim += ave_ssim
                print('#', end='')
            print('\nValidation, Loss: {:>.5f}, Loss (main): {:>.5f}, PSNR: {:>.5f}, SSIM: {:>.5f}'.format(\
                running_val_loss / num_batches_val, running_val_loss_main / num_batches_val, running_val_psnr / num_batches_val, running_val_ssim / num_batches_val))
            if writer != None:
                writer.add_scalar('total_loss_val', running_val_loss / num_batches_val, i_iter)
                writer.add_scalar('loss_main_val',  running_val_loss_main / num_batches_val, i_iter)
                writer.add_scalar('psnr_val', running_val_psnr / num_batches_val, i_iter)
                writer.add_scalar('ssim_val', running_val_ssim / num_batches_val, i_iter)
            model.train()

# --------------------------------------------------------------------------------------
# save the last one model
print('\nSave model ... (Epoch: {}, Iteration: {})'.format(i_epoch, i_iter))
model_dict = {'epoch': i_epoch, 'num_iter': i_iter, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
torch.save(model_dict, os.path.join(model_path, 'epoch_{}.pt'.format(i_iter + 1)))
# --------------------------------------------------------------------------------------
writer.flush() 
writer.close()
print('Training done!')