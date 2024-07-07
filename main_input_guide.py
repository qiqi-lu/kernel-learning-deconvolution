import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import os
import models.edsr as edsr
import models.srrln as srrln
import models.rcan as rcan
import models.rlsr as rlsr
import train
from utils import device_utils, dataset_utils
from utils import evaluation as eva
import tqdm

###################################################################################
gpu_id = 4
# ---------------------------------------------------------------------------------
random_seed = 7
num_workers = 6
fig_dir = os.path.join('outputs','figures')
checkpoint_dir = 'checkpoints'
# ---------------------------------------------------------------------------------
# model_name = 'edsr'
# model_name = 'srrln'
# model_name = 'rcan'

model_name = 'rlsr'
# multi_out = False
multi_out = True
n_iter = 2
n_blocks = 1

start_epoch = -1 
# start_epoch = 0 
weight_name = '0_9999'
batch_size = 1

learning_rate = 0.001
frac = 1.0
model_suffix = 'synth_iter_{}_block_{}_input_guide_sp_{}'.format(n_iter,n_blocks,frac)

# ---------------------------------------------------------------------------------
epochs = 100
save_every = 1
every_batch = 20
input_normalization = False

# ---------------------------------------------------------------------------------
# leanring rate schedule
lr_schedule = False
scheduler_cus = {}
scheduler_cus['lr'] = learning_rate
scheduler_cus['every'] = 5000
scheduler_cus['rate'] = 0.5

###################################################################################
print('='*98)
# set random seed
torch.manual_seed(random_seed)
# get device
device = device_utils.get_device(gpu_id=gpu_id)

###################################################################################
# Data 
###################################################################################
if input_normalization == True:
    print('> Input Normalization (on)')
    mean_normlize = np.array([0.4488,0.4371,0.4040])
    std_normlize  = np.array([1.0,1.0,1.0])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_normlize,std=std_normlize,inplace=True),
        ])
    data_transform_back = transforms.Compose([
        transforms.Normalize(mean=-mean_normlize/std_normlize,std=1.0/std_normlize),
        ])
if input_normalization == False:
    print('> Input Normalization (off)')
    data_transform = transforms.ToTensor()
    data_transform_back = None

# ---------------------------------------------------------------------------------
training_data_txt = os.path.join('data','raw','cyto_potable_microscope','train_txt') 
dir_hr    = os.path.join('data','raw','cyto_potable_microscope','data1')
dir_synth = os.path.join('data','raw','cyto_potable_microscope','data_synth','train','sf_4_k_2.0_gaussian_mix_ave')

# ---------------------------------------------------------------------------------
# Training data
# training_data   = dataset_utils.CytoDataset(txt_file=training_data_txt,root_dir=dir_hr,\
#                 transform=data_transform,id_range=[0,100000])

training_data = dataset_utils.CytoDataset_synth(txt_file=training_data_txt,dir_hr=dir_hr,\
    dir_synth=dir_synth,transform=data_transform,id_range=[0,100000])  

train_dataloader = DataLoader(dataset=training_data,batch_size=batch_size,\
    shuffle=True,num_workers=num_workers)

# ---------------------------------------------------------------------------------
# Validation data
# validation_data = dataset_utils.CytoDataset(txt_file=training_data_txt,root_dir=dir_hr,\
#                 transform=data_transform,id_range=[100000,120000])

# valid_dataloader = DataLoader(dataset=validation_data,batch_size=batch_size,\
#     num_workers=num_workers)

###################################################################################
# Model
###################################################################################
if model_name == 'edsr':
    model = edsr.EDSR(scale=4,n_colors=3,n_resblocks=16,n_features=128,kernel_size=3,\
                res_scale=0.1).to(device)

if model_name == 'srrln':
    model = srrln.SRRLN(scale=4,in_channels=3,n_features=4,kernel_size=3).to(device)

if model_name == 'rcan':
    model = rcan.RCAN(scale=4,n_colors=3,n_resgroups=5,n_resblocks=10,n_features=64,\
        kernel_size=3,reduction=16,res_scale=1.0).to(device)

if model_name == 'rlsr':
    model = rlsr.RLSR(scale=4,in_channels=3,n_features=16,n_blocks=n_blocks,n_iter=n_iter,\
        multi_output=multi_out,bias=True,inter=False,input_guide=True).to(device)

# ---------------------------------------------------------------------------------
# save graph
# print(model)
model_path = os.path.join(checkpoint_dir,'{}_bs_{}_lr_{}_{}'.\
    format(model_name,batch_size,learning_rate,model_suffix))
print('> Save model to ' + model_path)
writer = SummaryWriter(os.path.join(model_path,'log')) # TensorBoard writer

sample_one = next(iter(train_dataloader))
writer.add_graph(model,sample_one['lr'].to(device))

eva.count_parameters(model)

###################################################################################
# optimization
###################################################################################
loss_fn   = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=scheduler_cus['every'],\
    gamma=scheduler_cus['rate'])

# ---------------------------------------------------------------------------------
# load checkpoint
if start_epoch > -1:
    checkpoint_path = os.path.join(model_path,'epoch_{}.pt'.format(weight_name))
    print('> Reload model parameters (from ' + checkpoint_path + ')')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
for t in range(start_epoch + 1, epochs):
    print('-'*98, '\n Echo {} | Batch size {} \n'.format(t+1, batch_size),'-'*98)
    num_batches = len(train_dataloader)
    model.train()
    ave_ssim, ave_psnr = 0, 0

    pbar = tqdm.tqdm(total=num_batches,desc='TRAINING',leave=True,ncols=150) # processing bar

    # training
    for batch, sample in enumerate(train_dataloader):
        i = batch + t*num_batches
        x, y =  sample['lr'].to(device), sample['hr'].to(device)

        # Compute prediction error
        dvs, pred = model(x)

        if multi_out == True:
            loss1 = torch.abs(torch.mean(dvs)-1.0)
            loss2 = loss_fn(pred,y.repeat(n_iter,1,1,1,1))
        else:
            loss1 = torch.abs(torch.mean(dvs)-1.0)
            loss2 = loss_fn(pred,y)
        loss = frac*loss1 + loss2
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        # custom learning rate scheduler
        elif scheduler_cus is not None:
            if i % scheduler_cus['every'] == 0:
                for g in optimizer.param_groups:
                    g['lr'] = scheduler_cus['lr']*(scheduler_cus['rate']**(i//scheduler_cus['every']))
        
        # Metrics
        if data_transform_back is not None: 
            pred, y, x = data_transform_back(pred), data_transform_back(y), data_transform_back(x)

        if multi_out == True:
            out = pred[-1]
        else:
            out = pred

        if i % every_batch == 0:
            ave_ssim, ave_psnr = eva.measure(pred=train.rescale(out),y=train.rescale(y),x=train.rescale(x))

            if writer != None:
                writer.add_scalar('loss', loss_fn(out,y), i)
                writer.add_scalar('psnr',ave_psnr, i)
                writer.add_scalar('ssim',ave_ssim, i)
        
        if i % 1000 == 0:
            if writer != None:
                permute = [2,1,0]
                writer.add_images('image_x_batch',\
                    torch.nn.functional.interpolate(x[:,permute],scale_factor=4,mode='nearest'),i)
                writer.add_images('image_pred_batch',out[:,permute],i)
                writer.add_images('image_gt_batch',y[:,permute],i)

        pbar.set_postfix_str(s='loss: {:>7f}, PSNR: {:>.4f}, SSIM: {:>.4f}, lr: {:>.7f}'\
            .format(loss.item(),ave_psnr,ave_ssim,optimizer.param_groups[0]['lr']))
        pbar.update(1)

        # save model and relative information
        if (i+1) % 10000 == 0:
            model_dict = {  'epoch': t,
                            'num_iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}
            torch.save(model_dict,os.path.join(model_path,'epoch_{}_{}.pt'.format(t,i+1)))
            print('> Save model ...')

    pbar.close() 



    # loss_val = train.validation(dataloder=valid_dataloader,model=model,loss_fn=loss_fn,\
    #         device=device,data_transform=data_transform_back)

writer.flush() 
writer.close()

print('Done!')