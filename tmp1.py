import torch
import numpy as np
import skimage.io as skio
import utils.evaluation as eva

img_nf, img_n = [], []
for i in range(1):
    i = i+1
    path_img_nf = f"F:\\Datasets\\RLN\SimuMix3D_256s\\raw_psf_63_gauss_0_poiss_0_sf_1_ratio_1\\{i}.tif"
    path_img_n  = f"F:\\Datasets\\RLN\SimuMix3D_256s\\raw_psf_63_gauss_0.5_poiss_1_sf_1_ratio_1\\{i}.tif"
    img_nf.append(skio.imread(path_img_nf).astype(np.float32))
    img_n.append(skio.imread(path_img_n).astype(np.float32))

img_nf = np.array(img_nf)
img_n  = np.array(img_n)
print(img_nf.shape)

def cal_psnr(x, y):
    # need 3D input
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())

def cal_ssim(x, y):
    # need 3D input
    if y.shape[0] >= 7: # the size of the filter in SSIM is at least 7
        return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
            multichannel=False, channle_axis=None, version_wang=False)
    else:
        return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
            multichannel=True, channle_axis=0, version_wang=False)

out = eva.measure_3d(img_nf[...,None], img_n[...,None])
print(out)
print(cal_psnr(img_n[0], img_nf[0]))
print(cal_ssim(img_n[0], img_nf[0]))
print(np.mean((img_nf-img_n)**2))
print(np.mean(img_nf-img_n*np.log(img_nf+1e-8)))

