import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os, time
import methods.deconvolution as dcv
import utils.evaluation as eva
import torch

# ================================================================================================================
dataset_name = 'SimuBeads3D'
fig_path     = os.path.join('outputs', 'figures', 'simubeads3d')

# dataset_path = os.path.join('F:', os.sep, 'Datasets', 'RLN', dataset_name)
dataset_path = os.path.join('data', 'RLN', dataset_name)
# psf_path     = os.path.join(dataset_path, 'PSF_diSPIM.tif')
# PSF          = io.imread(psf_path).astype(np.float32)

data_id = [1,2,3,4,5]
# data_id = [1]

for i in data_id:
    print(i)
    data_gt_path   = os.path.join(dataset_path, 'ground_truth')
    data_blur_path = os.path.join(dataset_path, 'input_noise_0')

    data_gt   = io.imread(os.path.join(data_gt_path,   f'{i}.tif')).astype(np.float32)
    data_blur = io.imread(os.path.join(data_blur_path, f'{i}.tif')).astype(np.float32)

    # print('GT: {}, Input: {}, PSF: {}'.format(data_gt.shape, data_blur.shape, PSF.shape))

    for k in [1,2,3]:
        data_gt_rot   = np.rot90(data_gt, k, axes=(1,2))
        data_blur_rot = np.rot90(data_blur, k, axes=(1,2))

        io.imsave(fname=os.path.join(data_gt_path, f'{i}_1_{k}.tif'),  arr=data_gt_rot, check_contrast=False)
        io.imsave(fname=os.path.join(data_blur_path, f'{i}_1_{k}.tif'),  arr=data_blur_rot, check_contrast=False)

    data_gt_flip   = np.flip(data_gt, axis=2)
    data_blur_flip = np.flip(data_blur, axis=2)

    for k in [0, 1, 2, 3]:
        data_gt_rot   = np.rot90(data_gt_flip, k, axes=(1,2))
        data_blur_rot = np.rot90(data_blur_flip, k, axes=(1,2))

        io.imsave(fname=os.path.join(data_gt_path, f'{i}_2_{k}.tif'),  arr=data_gt_rot, check_contrast=False)
        io.imsave(fname=os.path.join(data_blur_path, f'{i}_2_{k}.tif'),  arr=data_blur_rot, check_contrast=False)
    


