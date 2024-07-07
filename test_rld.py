import os, torch, sys
import skimage.io as io
import utils.image_process as ip
import methods.rld as rld
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from utils import dataset_utils
# -----------------------------------------------------------------------------------
# Choose data set
# data_set_name = 'lena'
data_set_name = 'tinymicro_synth'
# data_set_name = 'tinymicro_real'
# data_set_name = 'lung3_synth'
# data_set_name = 'biosr_real'
# data_set_name = 'msi_synth'

input_normalization = False
num_sample_used_test = 100
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
else:
    data_transform = transforms.ToTensor()
    data_transform_back = None

# -----------------------------------------------------------------------------------
if data_set_name == 'tinymicro_synth':
    # TinyMicro (synth)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization = (False, False)
    in_channels = 3
    init_mode = 'bicubic'
    data_range = 255

if data_set_name == 'tinymicro_real':
    # TinyMicro (real)
    hr_root_path = os.path.join('data', 'TinyMicro', 'data')
    lr_root_path = os.path.join('data', 'TinyMicro', 'data')

    hr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join('data', 'TinyMicro', 'test_txt', 'lr.txt')
    normalization = (False, False)
    in_channels = 3
    init_mode = 'net'
    data_range = 255

if data_set_name == 'biosr_real':
    pass
if data_set_name == 'lung3_synth':
    # Lung3 (synth)
    if sys.platform == 'win32': root_path = os.path.join('F:', os.sep, 'Datasets')
    if sys.platform == 'linux' or sys.platform == 'linux2': root_path = 'data'
    hr_root_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939')
    lr_root_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'data_synth', 'test', 'sf_4_k_2.0_gaussian_0.03_ave')

    hr_txt_file_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'hr.txt') 
    lr_txt_file_path = os.path.join(root_path, 'Lung3', 'manifest-41uMmeOh151290643884877939', 'test_txt', 'lr.txt') 
    normalization = (False, True)
    in_channels = 1
    init_mode = 'bicubic'
    data_range = 1.0

if data_set_name == 'msi_synth':
    pass

if data_set_name in ['tinymicro_synth', 'tinymicro_real', 'lung3_synth', 'biosr_real', 'msi_synth']:
    test_data = dataset_utils.SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
            hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
            transform=data_transform, normalization=normalization)

    device = torch.device("cpu")
    id_data = 100
    ds = test_data[id_data]
    x, img_gt = torch.unsqueeze(ds['lr'], 0).to(device), torch.unsqueeze(ds['hr'], 0).to(device)
    if in_channels == 3: img_gt = img_gt.numpy().transpose((0, 2, 3, 1))[0]
    if in_channels == 1: img_gt = img_gt.numpy().transpose((0, 2, 3, 1))[0, ..., 0]

if data_set_name == 'lena':
    img_path = os.path.join('data',data_set_name,'Lena.tif')
    # img_path = os.path.join('data','lena','LenaRGB.tif')
    img_gt = io.imread(img_path) / 255.0
# -----------------------------------------------------------------------------------------------------
eps = 0.0001
kernel = ip.gaussian_kernel(n=25, std=2.0)
RL_deconv  = rld.RLD(kernel=kernel, data_range=(0, 1.0))
Degra = ip.ImageDegradation2D(scale_factor=4, kernel=kernel, noise_mode='gaussian', std=0.03, down_sample_mode='ave',\
    ratio=1000)

img_blur = Degra.convolution(img_gt)
img_down = Degra.downsampling(img_blur)
img_down_n = Degra.add_noise(img_down)
img_init = ip.interpolation(img_down_n, scale_factor=4, mode='bicubic').to(torch.float32)
img_blur, img_down, img_down_n, img_init = \
    ip.to_image(img_blur), ip.to_image(img_down), ip.to_image(img_down_n), ip.to_image(img_init)

img_fp = RL_deconv.conv(img_init)
img_dv = img_init / (img_fp + eps)
img_up = RL_deconv.backward_proj(img_dv)
img_1  = img_init * img_up

img_deblur_10   = RL_deconv.decov(img_init, 10)
img_deblur_100  = RL_deconv.decov(img_init, 100)
img_deblur_1000 = RL_deconv.decov(img_init, 1000)
# -----------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=4, dpi=600, figsize=(2.4 * 4, 2.4 * 3), constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,0].imshow(img_gt, cmap='gray', vmin=0.0, vmax=1.0)
axes[0,1].imshow(img_blur, cmap='gray', vmin=0.0, vmax=1.0)
axes[0,2].imshow(img_down, cmap='gray', vmin=0.0, vmax=1.0)
axes[0,3].imshow(img_down_n, cmap='gray', vmin=0.0, vmax=1.0)

axes[1,0].imshow(img_init, cmap='gray', vmin=0.0, vmax=1.0)
axes[1,1].imshow(img_fp, cmap='gray', vmin=0.0, vmax=1.0)
axes[1,2].imshow(img_dv, cmap='seismic', vmin=0.0, vmax=2.0)
axes[1,3].imshow(img_up, cmap='seismic', vmin=0.8, vmax=1.2)

axes[2,0].imshow(img_1, cmap='gray', vmin=0.0, vmax=1.0)
axes[2,1].imshow(img_deblur_10, cmap='gray', vmin=0.0, vmax=1.0)
axes[2,2].imshow(img_deblur_100, cmap='gray', vmin=0.0, vmax=1.0)
axes[2,3].imshow(img_deblur_1000, cmap='gray', vmin=0.0, vmax=1.0)

axes[0,0].set_title('HR')
axes[0,1].set_title('Blurred (std=2.0)')
axes[0,2].set_title('Downsampling (ave)')
axes[0,3].set_title('LR (noise, std=0.03)')

axes[1,0].set_title('Init (bicubic)')
axes[1,1].set_title('FP')
axes[1,2].set_title('DV')
axes[1,3].set_title('update')

axes[2,0].set_title('Deconv_1')
axes[2,1].set_title('Deconv (iter=10)')
axes[2,2].set_title('Deconv (iter=100)')
axes[2,3].set_title('Deconv (iter=1000)')

save_to = os.path.join('outputs','figures', data_set_name)
if not os.path.exists(save_to): os.makedirs(save_to, exist_ok=True)
plt.savefig(os.path.join(save_to,'rld'))