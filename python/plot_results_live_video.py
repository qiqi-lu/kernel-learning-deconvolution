import utils.image_plot as utils_plot
import skimage.io as io
import numpy as np
import os, cv2
import matplotlib as mpl
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
dataset_name = 'zeroshotdeconvnet'
# id_timepoint = [0, 346, 609, 700, 770, 901]
# id_timepoint = [0, 346]
id_timepoint = range(0, 1000, 4)
interval = 10 # s

# methods = ['raw', 'traditional', 'gaussian', 'butterworth',\
#     'wiener_butterworth','kernelnet']
# methods = ['raw', 'wiener_butterworth','kernelnet']
# methods = ['raw', 'kernelnet']
methods = ['kernelnet']

# ------------------------------------------------------------------------------
path_specimen = os.path.join('outputs', 'figures', dataset_name, 'Mitosis')
fig_path_data = [os.path.join(path_specimen, '642'),\
                 os.path.join(path_specimen, '560')]
dataset_path =\
    ['F:\\Datasets\\ZeroShotDeconvNet\\3D time-lapsing data_LLSM_Mitosis_H2B\\642',\
    'F:\\Datasets\\ZeroShotDeconvNet\\3D time-lapsing data_LLSM_Mitosis_Mito\\560']

with open(os.path.join(dataset_path[0], 'raw.txt')) as f:
    raw_txt = f.read().splitlines()

print('load result from :', fig_path_data)
Nc, Nt, Nmeth = len(dataset_path), len(id_timepoint), len(methods)
print('Num of channel: {}, Num of time point: {}, num of methods: {}'\
    .format(Nc, Nt, Nmeth))

# ------------------------------------------------------------------------------
img_single = io.imread(os.path.join(dataset_path[0], 'raw', raw_txt[0]))
Nz, Ny, Nx = img_single.shape

img_single_inter = utils_data.interp(img_single, ps_xy=92.6, ps_z=200)
Nz_inter, Ny, Nx = img_single_inter.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

center_xy, range_xy = Nz//2, 10
center_xz, range_xz = Ny//2, 10

for method in methods:
    print(method)
    video_xy = cv2.VideoWriter(os.path.join(path_specimen,\
        f'image_restored_{method}_xy.mp4'), fourcc, 1, (Nx, Ny))
    video_xz = cv2.VideoWriter(os.path.join(path_specimen,\
        f'image_restored_{method}_xz.mp4'), fourcc, 1, (Nx, Nz_inter))

    for i in id_timepoint:
        img = []
        for id_channel in range(len(fig_path_data)):
            fig_path_sample = os.path.join(fig_path_data[id_channel],\
                f'sample_{i}')

            print('load', fig_path_sample)
            if method == 'raw':
                img.append(io.imread(os.path.join(dataset_path[id_channel],\
                    'raw', raw_txt[i])))
            elif method == 'kernelnet':
                img.append(io.imread(os.path.join(fig_path_sample, method,\
                    # 'y_pred_all.tif'))[-1])
                    'y_pred_all.tif'))) # uint16
            else:
                img.append(io.imread(os.path.join(fig_path_sample, method,\
                    'deconv.tif')))
        img = np.transpose(np.array(img), axes=(1, 2, 3, 0)) # (Nz, Ny, Nx, Nc)

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
        if method == 'raw':
            vmin = [0, 0]
            vmax = [np.percentile(img[..., 0], 99.99),\
                    np.percentile(img[..., 1], 99.9)]
        else:
            vmin = [0, 0]
            vmax = [np.percentile(xy_plane[..., 0], 99.5),\
                    np.percentile(xy_plane[..., 1], 99.5)]

        cmaps = [mpl.colormaps['gray'], mpl.colormaps['afmhot']]
        dict_img = {'cmaps': cmaps, 'vmin':vmin, 'vmax': vmax, 'rgb_type':'bgr'}
        xy_plane = utils_plot.render(xy_plane, **dict_img)
        zy_plane = utils_plot.render(xz_plane, **dict_img)
        # ----------------------------------------------------------------------
        video_xy.write(image=xy_plane)
        video_xz.write(image=zy_plane)

    cv2.destroyAllWindows()
    video_xy.release()
    video_xz.release()

