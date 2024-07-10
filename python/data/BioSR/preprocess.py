'''
Preprocessing
Read image and convert to tiff file.
'''

import read_mrc as imread
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import skimage.measure as skim

# ------------------------------------------------------------------------------
# name_specimen, upsampling, scale_factor = 'CCPs', 2, 1
# name_specimen, upsampling, scale_factor = 'F-actin', 2, 1
# name_specimen, upsampling, scale_factor = 'Microtubules2', 2, 1
name_specimen, upsampling, scale_factor = 'F-actin_Nonlinear', 3, 1
noise_level = 9

# ------------------------------------------------------------------------------
path_dataset_raw = os.path.join('original', name_specimen)
path_save_gt  = os.path.join(name_specimen, f'gt_sf_{scale_factor}')
path_save_raw = os.path.join(name_specimen, f'raw_noise_{noise_level}')

print('save to:', path_save_gt, '|', path_save_raw)

for path in [path_save_gt, path_save_raw]:
    if os.path.exists(path) == False: os.makedirs(path, exist_ok=True)

path_cells_txt = os.path.join(path_dataset_raw, 'cells.txt')
with open(path_cells_txt) as f: cells = f.read().splitlines()
num_cells = len(cells)

print('Number of cells: ', num_cells)

# ------------------------------------------------------------------------------
# data processing
# ------------------------------------------------------------------------------
for i, id_cell in enumerate(cells):
    path_img = os.path.join(path_dataset_raw, id_cell)
    print(path_img)

    # load WF and GT image
    _, img_raw = imread.read_mrc(filename=\
        os.path.join(path_img, f'RawSIMData_level_0{noise_level}.mrc'))
    try:
        _, img_gt = imread.read_mrc(filename=\
            os.path.join(path_img, 'SIM_gt_a.mrc'))
    except:
        _, img_gt = imread.read_mrc(filename=\
            os.path.join(path_img, 'SIM_gt.mrc'))

    # A set of SIM image was average out to a wide-field images (Qiao, 2021).
    img_raw = np.mean(img_raw, axis=-1) # [Ny, Nx]

    # average pooling gt
    img_gt = np.squeeze(img_gt) # [Ny, Nx]
    img_gt = skim.block_reduce(img_gt, block_size=\
        (int(upsampling/scale_factor),)*2, func=np.mean)
    
    # rescale to have same average intensity
    ave_intensity = 100.0
    intensity_sum = ave_intensity * np.prod(img_raw.shape)
    img_gt  = img_gt  / img_gt.sum()  * intensity_sum
    img_raw = img_raw / img_raw.sum() * intensity_sum

    # Ny, Nx = img_gt.shape
    # if (Ny%2) == 0:
    #     img_raw = img_raw[:-1, :-1]
    #     img_gt  = img_gt[:-1, :-1]

    print('raw : {}, gt: {}'.format(img_raw.shape, img_gt.shape))

    # save image
    skio.imsave(fname=os.path.join(path_save_gt,  f'{i+1}.tif'), arr=img_gt,\
        check_contrast=False)
    skio.imsave(fname=os.path.join(path_save_raw, f'{i+1}.tif'), arr=img_raw,\
        check_contrast=False)
 
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.5, 2.5), dpi=600,\
    constrained_layout=True)

img_gt  = skio.imread(os.path.join(path_save_gt, '1.tif'))
img_raw = skio.imread(os.path.join(path_save_raw, '1.tif'))

axes[1].imshow(img_gt,  cmap='gray', vmin=0.0, vmax=img_gt.max()*0.7)
axes[0].imshow(img_raw, cmap='gray', vmin=0.0, vmax=img_gt.max()*0.7)
axes[2].plot(img_gt[100, 100:200])
axes[2].plot(img_raw[100, 100:200])
axes[0].set_title('WF')
axes[1].set_title('GT')

plt.savefig('example_image.png')