import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import utils.image_plot as utils_plot
import matplotlib as mpl
from skimage.measure import profile_line
import matplotlib.patches as patches
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
dataset_name = 'zeroshotdeconvnet'
id_sample    = 0
methods = ['raw', 'traditional', 'gaussian', 'butterworth',\
           'wiener_butterworth','kernelnet_ss','kernelnet']
# methods = ['raw', 'gaussian', 'wiener_butterworth', 'kernelnet_ss',\
    # 'kernelnet']

path_specimen = os.path.join('outputs', 'figures', dataset_name, 'Mitosis')

path_fig_data =\
    [os.path.join('outputs', 'figures', dataset_name, 'Mitosis', '642'),\
     os.path.join('outputs', 'figures', dataset_name, 'Mitosis', '560')]
path_root = 'F:\\Datasets\\ZeroShotDeconvNet\\'
path_dataset =\
    [path_root+'3D time-lapsing data_LLSM_Mitosis_H2B\\642',\
     path_root+'3D time-lapsing data_LLSM_Mitosis_Mito\\560']

with open(os.path.join(path_dataset[0], 'raw.txt')) as f:
    raw_txt = f.read().splitlines()

# ------------------------------------------------------------------------------
# load results
print('>> Load result from :', path_fig_data)
imags_mc = []

for id_channel in range(len(path_fig_data)):
    path_fig_sample = os.path.join(path_fig_data[id_channel],\
        f'sample_{id_sample}')
    print('>> Load', path_fig_sample)

    imgs = []
    # raw
    imgs.append(io.imread(os.path.join(path_dataset[id_channel],\
        methods[0], raw_txt[id_sample])))
    # conventional methods
    for meth in methods[1:-2]:
        imgs.append(io.imread(os.path.join(path_fig_sample, meth,\
            'deconv.tif')))
    # KLD-ss
    imgs.append(io.imread(os.path.join(path_fig_sample, methods[-2],\
        'y_pred_all.tif'))[-1])
    # KLD
    imgs.append(io.imread(os.path.join(path_fig_sample, methods[-1],\
        'y_pred_all.tif')))

    imags_mc.append(imgs)
imags_mc = np.array(imags_mc)

Nc, Nmeth, Nz, Ny, Nx = imags_mc.shape
print('Num of channel: {}, num of methods: {}, image shape: {}'\
    .format(Nc, Nmeth, (Nz, Ny, Nx)))

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
# define color
nr, nc = 3, Nmeth
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=600, figsize=(3*nc, 3*nr),\
    constrained_layout=True)
for ax in axes.ravel(): ax.set_axis_off()

line_start = (197, 83)
line_end   = (210, 98)

pos  = (175, 195)
pos  = (120, 155)
size = (60, 60)

dict_line = {'color': 'white', 'linewidth': 1}

center_xy, center_xz = 60, 150
range_xy,  range_xz  = 10, 10

for imeth in range(Nmeth):
    img = imags_mc[:, imeth]
    img = np.transpose(img, axes=(1, 2, 3, 0))

    # xy plane
    # --------------------------------------------------------------------------
    xy_plane = np.max(\
        img[center_xy - range_xy : center_xy + range_xy], axis=0)

    # --------------------------------------------------------------------------
    # xz plane
    xz_plane = np.max(\
        img[:, (center_xz - range_xz) : (center_xz + range_xz), :], axis=1)
    
    tmp = []
    for i in range(Nc):
        tmp.append(utils_data.interp(xz_plane[..., i], ps_xy=92.6, ps_z=200))
    xz_plane = np.transpose(np.array(tmp), axes=(1, 2, 0))
    # --------------------------------------------------------------------------
    if imeth == 0:
        vmin = [0, 0]
        vmax = [np.percentile(xy_plane[..., 0], 99.99), \
                np.percentile(xy_plane[..., 1], 99.9)]
    else:
        vmin = [0, 0]
        vmax = [np.percentile(xy_plane[..., 0], 99.5),\
                np.percentile(xy_plane[..., 1], 99.5)]

    # --------------------------------------------------------------------------
    cmaps = [mpl.colormaps['gray'], mpl.colormaps['afmhot']]
    dict_img = {'cmaps': cmaps, 'vmin':vmin, 'vmax': vmax}
    axes[0, imeth].imshow(utils_plot.render(xy_plane, **dict_img))
    axes[1, imeth].imshow(utils_plot.render(xz_plane, **dict_img))
    axes[0, imeth].text(10, 20, methods[imeth], color='white')

    axes[0, 0].plot((line_start[0], line_end[0]), (line_start[1], line_end[1]),\
        **dict_line)

    # --------------------------------------------------------------------------
    # box
    id_channel = 0
    patch = patches.Rectangle(xy=pos, width=size[1], height=size[0],\
        fill=False, edgecolor='white')
    axes[0, 0].add_patch(patch)
    axes[2, imeth].imshow(xy_plane[pos[1]:pos[1]+size[1],\
        pos[0]:pos[0]+size[0], id_channel], vmax=vmax[id_channel],\
        vmin=vmin[id_channel], cmap=cmaps[id_channel])

plt.savefig(os.path.join(path_specimen, \
    f'image_restored_{methods[imeth]}_compare_{id_sample}.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_specimen, \
    f'image_restored_{methods[imeth]}_compare_{id_sample}.svg'))

# ------------------------------------------------------------------------------
nr, nc = 1, Nmeth
fig, axes = plt.subplots(nrows=nr, ncols=nc, dpi=300, figsize=(3*nc, 3*nr),\
    constrained_layout=True)

id_channel = 0
dict_profile = {'linewidth': 0.5}

line_start = (line_start[1],line_start[0])
line_end   = (line_end[1], line_end[0])

for imeth in range(Nmeth):
    img = imags_mc[id_channel, imeth]
    xy_plane = np.max(\
        img[center_xy - range_xy : center_xy + range_xy],axis=0)

    profile = profile_line(xy_plane, line_start, line_end, linewidth=1)
    axes[imeth].plot(profile, label= methods[imeth], color='#2A629A',\
        **dict_profile)

io.imsave(fname=os.path.join(path_specimen, 'xy.tif'), arr=xy_plane,\
    check_contrast=False)

for ax in axes.ravel():
    ax.tick_params(direction='in')

    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    ax.set_ylabel('Intensity (AU)')
    ax.set_xlabel('Distance (pixel)')

plt.legend(fontsize='xx-small')
plt.savefig(os.path.join(path_specimen, 'img_restored_profile.png'))
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(os.path.join(path_specimen, 'img_restored_profile.svg'))