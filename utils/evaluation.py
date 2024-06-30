import skimage.metrics as skim
from utils import dataset_utils
import numpy as np
from pytorch_msssim import ms_ssim
import torch

def SSIM(img_true, img_test, data_range=1.0, multichannel=False,\
    channle_axis=None,version_wang=False):
    '''
    Structrual similarity for an `single-channel/multi-channle 2D` or
    `single-channel 3D` image. 
    
    Args:
    - img_true (array): ground truth.
    - img_test (array): predicted image.
    - data_range (float, int): image value range.
    - version_wang (bool): use parameter used in Wang's paper.
    '''
    # if len(img_true.shape) == 2: np.expand_dims(img_true, axis=-1)
    # if len(img_test.shape) == 2: np.expand_dims(img_test, axis=-1)

    if version_wang == False:
        ssim = skim.structural_similarity(im1=img_true, im2=img_test,\
            data_range=data_range, channel_axis=channle_axis)

    if version_wang == True:
        ssim = skim.structural_similarity(im1=img_true, im2=img_test,\
            multichannel=multichannel, data_range=data_range,\
            channel_axis=channle_axis, gaussian_weights=True, sigma=1.5,\
            use_sample_covariance=False)
    return ssim

def MSE(img_true, img_test):
    '''Mean Square error for one subject.
    '''
    mse = np.mean((img_test - img_true)**2)
    return mse

def RMSE(x, y):
    '''
    - y: groud truth
    '''
    rmse = np.mean(np.square(y-x))/np.mean(np.square(y))*100
    return rmse

def PSNR(img_true, img_test, data_range=255):
    '''
    Args:
    - img_true (array): ground truth.
    - img_test (array): predicted image.
    - data_range (float, int): image value range.
    '''
    if len(img_true.shape) == 2: np.expand_dims(img_true, axis=-1)
    if len(img_test.shape) == 2: np.expand_dims(img_test, axis=-1)
    
    psnr = skim.peak_signal_noise_ratio(image_true=img_true, image_test=img_test, data_range=data_range)
    return psnr

def SNR(img_true, img_test, type=0):
    if type == 0:
        img_true_ss = np.sum(np.square(img_true))
        error_ss = np.sum(np.square(img_true - img_test))
    if type == 1:
        img_true_ss = np.var(img_true)
        error_ss = np.var(img_test - img_true)
    snr = 10* np.log10(img_true_ss/error_ss)
    return snr

def NCC(img_true, img_test):
    mean_true = img_true.mean()
    mean_test = img_test.mean()
    sigma_true = img_true.std()
    sigma_test = img_test.std()
    NCC = np.mean((img_true-mean_true)*(img_test-mean_test)/(sigma_true*sigma_test))
    return NCC

def NRMSE(img_true, img_test):
    xmax, xmin  = np.max(img_true), np.min(img_true)
    rmse = np.sqrt(np.mean(np.square(img_test - img_true)))
    nrmse = rmse / (xmax - xmin)
    return nrmse

def MSSSIM(img_true, img_test, data_range=255):
    img_true = torch.Tensor(img_true)
    img_test = torch.Tensor(img_test)
    if len(img_true.shape) == 3: img_true = img_true[None]
    if len(img_test.shape) == 3: img_test = img_test[None]
    img_true = torch.transpose(img_true, dim0=-1, dim1=1)
    img_test = torch.transpose(img_test, dim0=-1, dim1=1)
    msssim = ms_ssim(img_true, img_test, data_range=data_range, size_average=False)
    return msssim

def measure(img_true, img_test, data_range=255):
    '''
    Measure metrics of each sample (along the 0 axis) and average.
    Args:
    - img_true (tensor): ground truth.
    - img_test (tensor): test image.
    - data_range (int, optional): The data range of the input images. Default: 255.
    Returns:
    - ave_ssim (float): average ssim.
    - ave_psnr (float): average psnr.
    '''
    ssim, psnr = [], []
    if not isinstance(img_true, np.ndarray):
        ToNumpy = dataset_utils.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        if len(img_true.shape) == 4: ssim.append(SSIM(img_true[i], img_test[i], data_range=data_range))
        if len(img_true.shape) == 5: 
            # ssim.append(SSIM(img_true[i,...,-1], img_test[i,...,-1], data_range=data_range, multichannel=False, channle_axis=None, version_wang=False))
            ssim.append(0)
        psnr.append(PSNR(img_true[i], img_test[i], data_range=data_range))
    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr

def measure_3d(img_true, img_test, data_range=None):
    ssim, psnr = [], []

    if not isinstance(img_true, np.ndarray):
        ToNumpy = dataset_utils.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        # if data_range is not None:
        #     data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        y, x = img_true[i, ..., 0], img_test[i, ..., 0]
        if data_range == None: data_range = y.max() - y.min()
        if y.shape[0] >= 7:
            ssim.append(SSIM(img_true=y, img_test=x, data_range=data_range,\
                multichannel=False, channle_axis=None, version_wang=False))
        else:
            ssim.append(SSIM(img_true=y, img_test=x, data_range=data_range,\
                multichannel=True, channle_axis=0, version_wang=False))
        psnr.append(PSNR(img_true=y, img_test=x, data_range=data_range))

    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr

def measure_2d(img_true, img_test, data_range=None):
    ssim, psnr = [], []

    # convert to numpy array
    if not isinstance(img_true, np.ndarray):
        ToNumpy = dataset_utils.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        # if data_range is not None:
        #     data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        y, x = img_true[i, ..., 0], img_test[i, ..., 0]
        if data_range == None: data_range = y.max() - y.min()
        ssim.append(SSIM(img_true=y, img_test=x, data_range=data_range,\
            multichannel=True, channle_axis=0, version_wang=False))
        psnr.append(PSNR(img_true=y, img_test=x, data_range=data_range))

    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr

def metrics_batch(img_true,img_test,data_range=255):
    img_true = dataset_utils.tensor2rgb(img_true)
    img_test = dataset_utils.tensor2rgb(img_test)
    ssim, psnr = [], []

    for i in range(len(img_true)):
        ssim.append(SSIM(img_true[i],img_test[i],data_range=data_range))
        psnr.append(PSNR(img_true[i],img_test[i],data_range=data_range))
    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr

def count_parameters(model):
    total_para = sum(p.numel() for p in model.parameters())
    trainbale_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Parameters: {:>10d}, Trainable Parameters: {:>10d}, Non-trainable Parameters: {:>10d}'\
        .format(total_para,trainbale_para,total_para-trainbale_para))

