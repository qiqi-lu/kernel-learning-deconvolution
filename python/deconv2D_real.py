'''
Use conventional deconvolution method to restore 3D image.
Requirements:
- Ground truth
- PSF
'''

import numpy as np
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva
from tabulate import tabulate as tabu

# ------------------------------------------------------------------------------
def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")

# ----------------------------------------------------------------------------
# dataset_name = 'F-actin_Nonlinear'
dataset_name = 'Microtubules2'

id_data = [0, 1, 2, 3, 4, 5, 6]
# id_data = [7, 8, 9, 10]
# id_data = [6]

scale_factor, std_gauss = 1, 9

# ----------------------------------------------------------------------------
enable_traditonal, enable_gaussian, enable_bw, enable_wb = 1, 0, 0, 0
num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 100, 30, 30, 2 
# num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 3, 30, 30, 2 

# ----------------------------------------------------------------------------
# load data
path_dataset  = os.path.join('F:', os.sep, 'Datasets', 'BioSR', dataset_name)
path_data_gt  = os.path.join(path_dataset, f'gt_sf_{scale_factor}')
path_data_raw = os.path.join(path_dataset, f'raw_noise_{std_gauss}')

with open(os.path.join(path_dataset, 'test.txt')) as f:
    test_txt = f.read().splitlines() 

PSF = io.imread(os.path.join('outputs', 'figures', dataset_name,\
    f'scale_{scale_factor}_gauss_{std_gauss}_poiss_1_ratio_1',\
    'kernels_bc_1_re_1',\
    'kernel_fp.tif'))
PSF = PSF.astype(np.float32)

# ------------------------------------------------------------------------------
# evaluation metrics
cal_ssim = lambda x, y:\
    eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min(),\
        version_wang=False)
cal_mse  = lambda x, y:\
    eva.PSNR(img_true=y, img_test=x,data_range=y.max() - y.min())
cal_ncc  = lambda x, y:\
    eva.NCC(img_true=y, img_test=x)
metrics  = lambda x :\
    [cal_mse(x, data_gt), cal_ssim(x, data_gt), cal_ncc(x, data_gt)]

# ------------------------------------------------------------------------------
DCV_trad = dcv.Deconvolution(PSF=PSF, bp_type='traditional', init='measured',\
    metrics=metrics, padding_mode='reflect')
# DCV_gaus = dcv.Deconvolution(PSF=PSF, bp_type='gaussian', init='measured',\
#     metrics=metrics)
# DCV_butt = dcv.Deconvolution(PSF=PSF, bp_type='butterworth', beta=0.01, n=10,\
#     res_flag=1, init='measured', metrics=metrics)
# DCV_wb = dcv.Deconvolution(PSF=PSF, bp_type='wiener-butterworth', alpha=0.005,\
#     beta=0.1, n=10, res_flag=1, init='measured', metrics=metrics)

# ------------------------------------------------------------------------------
for id in id_data:
    data_gt    = io.imread(os.path.join(path_data_gt, test_txt[id]))
    data_input = io.imread(os.path.join(path_data_raw, test_txt[id]))

    data_gt    = data_gt.astype(np.float32)
    data_input = data_input.astype(np.float32)

    PSF_align  = dcv.align_size(PSF, data_gt.shape)
    print('-'*80)
    print('load data from:', path_data_raw)
    print('GT: {}, Input: {}, PSF: {}'\
        .format(list(data_gt.shape), list(data_input.shape), list(PSF.shape)))

    # --------------------------------------------------------------------------
    # save result to path
    path_fig = os.path.join('outputs', 'figures', dataset_name.lower(),\
        f'scale_{scale_factor}_gauss_{std_gauss}_poiss_1_ratio_1',\
        f'sample_{id}')

    if not os.path.exists(path_fig): os.makedirs(path_fig)

    print('Save results to :', path_fig)

    # for meth in ['traditional', 'gaussian','butterworth','wiener_butterworth']:
    for meth in ['traditional']:
        path_meth = os.path.join(path_fig, meth)
        if not os.path.exists(path_meth): os.makedirs(path_meth)

    # --------------------------------------------------------------------------
    if enable_traditonal:
        out_trad = DCV_trad.deconv(data_input, num_iter=num_iter_trad,\
            domain='fft')
        bp_trad  = DCV_trad.PSF2
        out_gaus_metrics = DCV_trad.get_metrics() 

        io.imsave(fname=os.path.join(path_fig, 'traditional', 'deconv.tif'),\
            arr=out_trad, check_contrast=False)
        io.imsave(fname=os.path.join(path_fig, 'traditional', 'deconv_bp.tif'),\
            arr=bp_trad, check_contrast=False)
    
    # print('-'*80)
    # print(tabulate(out_gaus_metrics.transpose()))
    # print('-'*80)

    # # ----------------------------------------------------------------------------
    # if enable_gaussian:
    #     out_gaus    = DCV_gaus.deconv(data_input, num_iter=num_iter_gaus,\
    #         domain='fft')
    #     bp_gaus     = DCV_gaus.PSF2
    #     bp_gaus_otf = DCV_gaus.OTF_bp
    #     out_gaus_metrics = DCV_gaus.get_metrics()

    #     io.imsave(fname=os.path.join(path_fig, 'gaussian', 'deconv.tif'),\
    #         arr=out_gaus, check_contrast=False)
    #     io.imsave(fname=os.path.join(path_fig, 'gaussian', 'deconv_bp.tif'),\
    #         arr=bp_gaus, check_contrast=False)

    # # ----------------------------------------------------------------------------
    # if enable_bw:
    #     out_bw    = DCV_butt.deconv(data_input, num_iter=num_iter_bw,\
    #         domain='fft')
    #     bp_bw     = DCV_butt.PSF2
    #     bp_bw_otf = DCV_butt.OTF_bp
    #     out_bw_metrics = DCV_butt.get_metrics()

    #     io.imsave(fname=os.path.join(path_fig, 'butterworth', 'deconv.tif'),\
    #         arr=out_bw, check_contrast=False)
    #     io.imsave(fname=os.path.join(path_fig, 'butterworth', 'deconv_bp.tif'),\
    #         arr=bp_bw, check_contrast=False)

    # # ----------------------------------------------------------------------------
    # if enable_wb:
    #     out_wb    = DCV_wb.deconv(data_input, num_iter=num_iter_wb,\
    #         domain='fft')
    #     bp_wb     = DCV_wb.PSF2
    #     bp_wb_otf = DCV_wb.OTF_bp
    #     out_wb_metrics = DCV_wb.get_metrics()

    #     io.imsave(fname=os.path.join(path_fig, 'wiener_butterworth',\
    #         'deconv.tif'), arr=out_wb, check_contrast=False)
    #     io.imsave(fname=os.path.join(path_fig, 'wiener_butterworth',\
    #         'deconv_bp.tif'), arr=bp_wb, check_contrast=False)
