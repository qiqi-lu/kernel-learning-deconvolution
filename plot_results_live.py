import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import utils.image_plot as utils_plot
import matplotlib as mpl
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
dataset_name = 'zeroshotdeconvnet'
id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = [7, 8 , 9]
interval = 10 # s
# ------------------------------------------------------------------------------
# methods = ['raw', 'traditional', 'gaussian', 'butterworth',\
#            'wiener_butterworth','kernelnet']
# methods = ['raw', 'wiener_butterworth', 'kernelnet_ss', 'kernelnet']
methods = ['raw', 'kernelnet']

path_specimen = os.path.join('outputs', 'figures', dataset_name, 'Mitosis')

path_fig_data = [os.path.join(path_specimen, '642'),\
                 os.path.join(path_specimen, '560')]
path_root = 'F:\\Datasets\\ZeroShotDeconvNet\\'
path_dataset =\
    [path_root+'3D time-lapsing data_LLSM_Mitosis_H2B\\642',\
     path_root+'3D time-lapsing data_LLSM_Mitosis_Mito\\560']

with open(os.path.join(path_dataset[0], 'raw.txt')) as f:
    raw_txt = f.read().splitlines()

# ------------------------------------------------------------------------------
# load results
print('-'*80)
print('load result from :', path_fig_data)
img_deconv_mc = []

for id_channel in range(len(path_fig_data)):
    img_deconv = []
    for i in id_sample:
        path_fig_sample = os.path.join(path_fig_data[id_channel], f'sample_{i}')
        print('load', path_fig_sample)
        imgs = []
        # raw
        imgs.append(io.imread(os.path.join(path_dataset[id_channel],\
            methods[0], raw_txt[i])).astype(np.uint16))
        # # conventional methods
        # for meth in methods[1:-2]:
        #     imgs.append(io.imread(os.path.join(path_fig_sample, meth,\
        #         'deconv.tif')))
        # # KLD-ss
        # imgs.append(io.imread(os.path.join(path_fig_sample, methods[-2],\
        #     'y_pred_all.tif'))[-1])
        # KLD
        # imgs.append(io.imread(os.path.join(path_fig_sample, methods[-1],\
        #     'y_pred_all.tif'))[-1])
        imgs.append(io.imread(os.path.join(path_fig_sample, methods[-1],\
            'y_pred_all.tif')))
        img_deconv.append(imgs)
    img_deconv_mc.append(img_deconv)
img_deconv_mc = np.array(img_deconv_mc)

Nc, Nt, Nmeth, Nz, Ny, Nx = img_deconv_mc.shape
print('Num of channel: {}, Num of time point: {}, num of methods: {},\
    image shape: {}'.format(Nc, Nt, Nmeth, (Nz, Ny, Nx)))

# ------------------------------------------------------------------------------
# show image restored (merged)
# ------------------------------------------------------------------------------
# define color
nr, nc = 2, Nt
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes.ravel(): ax.set_axis_off()

center_xy, range_xy = Nz//2, 10
center_xz, range_xz = Ny//2, 10

for imeth in range(Nmeth):
    print('plot', methods[imeth])
    for it in range(Nt):
        img = img_deconv_mc[:, it, imeth]
        img = np.transpose(img, axes=(1, 2, 3, 0))

        # xy plane
        # ----------------------------------------------------------------------
        xy_plane = np.max(\
            img[center_xy - range_xy : center_xy + range_xy], axis=0)

        # ----------------------------------------------------------------------
        # xz plane
        xz_plane = np.max(\
            img[:, (center_xz - range_xz) : (center_xz + range_xz), :], axis=1)

        tmp = []
        for i in range(Nc):
            tmp.append(utils_data.interp(xz_plane[..., i],\
                ps_xy=92.6, ps_z=200))
        xz_plane = np.transpose(np.array(tmp), axes=(1, 2, 0))

        # ----------------------------------------------------------------------
        cmaps = [mpl.colormaps['gray'], mpl.colormaps['afmhot']]

        if imeth == 0:
            # vmin = [np.percentile(img[..., 0], 3),\
            #         np.percentile(img[..., 1], 3)]
            vmin = [0, 0]
            vmax = [np.percentile(img[..., 0], 99.99),\
                    np.percentile(img[..., 1], 99.9)]
        else:
            # vmin = [np.percentile(img[..., 0], 3),\
            #         np.percentile(img[..., 1], 3)]
            vmin = [0, 0]
            vmax = [np.percentile(xy_plane[..., 0], 99.5),\
                    np.percentile(xy_plane[..., 1], 99.5)]
        
        # vmin = [0, 0]
        # vmax = [np.percentile(xy_plane[..., 0], 99.5),\
        #         np.percentile(xy_plane[..., 1], 99.5)]

        dict_img = {'cmaps': cmaps, 'vmin':vmin, 'vmax': vmax}
        axes[0, it].imshow(utils_plot.render(xy_plane, **dict_img))
        axes[0, it].text(10, 20,\
            '{:>.1f} min'.format(interval*id_sample[it]/60), color='white')
        axes[1, it].imshow(utils_plot.render(xz_plane, **dict_img))

    plt.savefig(os.path.join(path_specimen,\
        f'image_restored_{methods[imeth]}.png'))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(os.path.join(path_specimen,\
        f'image_restored_{methods[imeth]}.svg'))


# ------------------------------------------------------------------------------
# outout to Amira struture
# ------------------------------------------------------------------------------
# print('>> save as Amira shape...')
# for imeth in range(Nmeth):
#     print(methods[imeth])
#     data_amira = img_deconv_mc[:,:,imeth,...]
#     data_amira = np.transpose(data_amira, (2,3,4,0,1))
#     io.imsave(fname=os.path.join(path_specimen, f'{methods[imeth]}.tif'),\
#         arr=data_amira, check_contrast=False)

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
# nr, nc = 2, Nt
# fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
#     constrained_layout=True)
# for ax in axes.ravel(): ax.set_axis_off()

# dict_img = {'cmap': 'gray', 'vmin':0}

# # show each channel image
# for ichannel in range(len(path_fig_data)):
#     print('Channel:', ichannel)
#     for imeth in range(Nmeth):
#         print('plot', methods[imeth])
#         for it in range(Nt):
#             img = img_deconv_mc[ichannel, it, imeth]

#             # axes[0, it].imshow(img[Nz//2], vmax=np.percentile(img, 99.99),\
#             #     **dict_img)
#             # axes[1, it].imshow(img[:,:,Nx//2], vmax=np.percentile(img, 99.99),\
#             #     **dict_img)

#             MIP_slice = 16
#             xy_plane = np.max(\
#                 img[Nz//2 - MIP_slice//2 : Nz//2 + MIP_slice//2], axis=0)
#             xz_plane = np.max(\
#                 img[:, Ny//2 - MIP_slice//2 : Ny//2 + MIP_slice//2, :], axis=1)

#             axes[0, it].imshow(xy_plane, vmax=np.percentile(xy_plane, 99.99),\
#                 **dict_img)
#             axes[1, it].imshow(xz_plane, vmax=np.percentile(xz_plane, 99.99),\
#                 **dict_img)
#             axes[0, it].text(10, 20,\
#                 '{:>.1f} min'.format(interval*id_sample[it]/60), color='white')

#         plt.savefig(os.path.join(path_fig_data[ichannel],\
#             f'image_restored_{methods[imeth]}.png'))
