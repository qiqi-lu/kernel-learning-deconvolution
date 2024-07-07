import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os

# ------------------------------------------------------------------------------
# dataset_name = 'Microtubule'
# dataset_name = 'Nuclear_Pore_complex'

dataset_name = 'Microtubule2'
# dataset_name = 'Nuclear_Pore_complex2'

# ------------------------------------------------------------------------------
fig_path = os.path.join('outputs', 'figures', dataset_name.lower())
if not os.path.exists(fig_path):\
    os.makedirs(fig_path, exist_ok=True)

# ------------------------------------------------------------------------------
# load data
path_dataset = os.path.join('F:', os.sep, 'Datasets', 'RCAN3D',\
    'Confocal_2_STED', dataset_name)
path_gt_txt  = os.path.join(path_dataset, 'gt.txt')
path_raw_txt = os.path.join(path_dataset, 'raw.txt')

with open(path_gt_txt) as f:  file_name_gt  = f.read().splitlines()
with open(path_raw_txt) as f: file_name_raw = f.read().splitlines()

# patch_enable = True
patch_enable = False

if patch_enable:
    # step, patch_size, N_step = 125, 128, 8
    step, patch_size, N_step = 500, 512, 2
    save_to_gt  = os.path.join(path_dataset, 'gt_512x512')
    save_to_raw = os.path.join(path_dataset, 'raw_512x512')
else:
    save_to_gt  = os.path.join(path_dataset, 'gt_1024x1024')
    save_to_raw = os.path.join(path_dataset, 'raw_1024x1024')

for path in [save_to_raw, save_to_gt]:
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)

print('load data from :', path_dataset)
print('number of data :', len(file_name_gt))
print('save data to :')
print(save_to_gt)
print(save_to_raw)

# ------------------------------------------------------------------------------
def preprocess(path, name_gt, name_raw):
    ave_intensity = 100.0

    data_gt  = io.imread(os.path.join(path, 'gt',  name_gt )).astype(np.float32)
    data_raw = io.imread(os.path.join(path, 'raw', name_raw)).astype(np.float32)

    print('Sample:', name_gt)
    print('GT: {}, Input: {}'.format(data_gt.shape, data_raw.shape))
    print(data_gt.mean(), data_raw.mean(), data_gt.sum()/data_raw.sum())

    n_pad = 1024
    data_gt  = np.pad(data_gt,  pad_width=((0,0), (0, n_pad-data_gt.shape[1]),\
        (0, n_pad-data_gt.shape[2])), mode='edge')
    data_raw = np.pad(data_raw, pad_width=((0,0), (0, n_pad-data_raw.shape[1]),\
        (0, n_pad-data_raw.shape[2])), mode='edge')
    
    # positive constriant (2, new version)
    data_gt  = np.maximum(data_gt, 0.0)
    data_raw = np.maximum(data_raw, 0.0)

    # normalization
    intensity_sum = ave_intensity * np.prod(data_raw.shape)
    data_gt  = data_gt/data_gt.sum()*intensity_sum
    data_raw = data_raw/data_raw.sum()*intensity_sum

    return data_gt, data_raw

# ------------------------------------------------------------------------------
# show example
# ------------------------------------------------------------------------------
id_data_show = 1
data_gt, data_raw = preprocess(path_dataset, file_name_gt[id_data_show],\
    file_name_raw[id_data_show])

Nz, Ny, Nx = data_gt.shape
# ------------------------------------------------------------------------------
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3* nc, 3* nr),\
    constrained_layout=True)
[ax.set_axis_off() for ax in axes[0:2, 0:2].ravel()]

dict_img = {'cmap':'gray', 'vmax':data_gt.max() * 0.6, 'vmin':0}

axes[0,0].set_title('GT (max={:.2f})'.format(data_gt.max()))
axes[0,1].set_title('RAW (max={:.2f})'.format(data_raw.max()))

axes[0,0].imshow(data_gt[Nz//2],  **dict_img)
axes[0,1].imshow(data_raw[Nz//2], **dict_img)
axes[0,2].plot(data_gt[Nz//2, 100, 50:500], 'red')
axes[0,2].plot(data_raw[Nz//2, 100, 50:500], 'green')

axes[1,0].imshow(data_gt[Nz//2+1],  **dict_img)
axes[1,1].imshow(data_raw[Nz//2+1], **dict_img)
axes[1,2].plot(data_gt[Nz//2+1, 100, 50:500], 'red')
axes[1,2].plot(data_raw[Nz//2+1, 100, 50:500], 'green')
plt.savefig(os.path.join(fig_path, 'data_check.png'))

# ------------------------------------------------------------------------------
if patch_enable == False:
    for i in range(len(file_name_gt)):
        print(file_name_gt[i])
        data_gt, data_raw = preprocess(path_dataset, file_name_gt[i],\
            file_name_raw[i])
        io.imsave(os.path.join(save_to_gt, file_name_gt[i]), arr=data_gt,\
            check_contrast=False)
        io.imsave(os.path.join(save_to_raw, file_name_gt[i]), arr=data_raw,\
            check_contrast=False)

if patch_enable == True:
    for i in range(len(file_name_gt)):
        print(file_name_gt[i])
        data_gt, data_raw = preprocess(path_dataset, file_name_gt[i],\
            file_name_raw[i])
        for m in range(N_step):
            for n in range(N_step):
                patch_gt    = data_gt[:, (0+step*m):(patch_size+step*m),\
                    (0+step*n):(patch_size+step*n)]
                patch_input = data_raw[:, (0+step*m):(patch_size+step*m),\
                    (0+step*n):(patch_size+step*n)]
                io.imsave(os.path.join(save_to_gt, f'{i}_{m}_{n}.tif'),\
                    arr=patch_gt, check_contrast=False)
                io.imsave(os.path.join(save_to_raw, f'{i}_{m}_{n}.tif'),\
                    arr=patch_input, check_contrast=False)

