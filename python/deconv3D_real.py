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

def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")

# ------------------------------------------------------------------------------
# dataset_name = 'Microtubule'
# dataset_name = 'Nuclear_Pore_complex'

dataset_name = 'Microtubule2'
# dataset_name = 'Nuclear_Pore_complex2'

id_data = [0, 1, 2, 3, 4, 5, 6]
# id_data = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ------------------------------------------------------------------------------
enable_traditonal, enable_gaussian, enable_bw, enable_wb = 1, 0, 0, 0
# num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 15, 30, 30, 2 
# num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 20, 30, 30, 2 # NPC
num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 15, 30, 30, 2 # MT

# ------------------------------------------------------------------------------
# load data
dataset_path  = os.path.join('F:', os.sep, 'Datasets', 'RCAN3D',\
    'Confocal_2_STED', dataset_name)
data_gt_path  = os.path.join(dataset_path, 'gt_1024x1024')
data_raw_path = os.path.join(dataset_path, 'raw_1024x1024')

with open(os.path.join(dataset_path, 'test_1024x1024.txt')) as f:
    test_txt = f.read().splitlines() 

if dataset_name in ['Microtubule', 'Nuclear_Pore_complex']:
    PSF = io.imread(os.path.join('outputs', 'figures', dataset_name,\
        'scale_1_noise_0_ratio_1', 'kernel_fp.tif')).astype(np.float32)

if dataset_name in ['Microtubule2', 'Nuclear_Pore_complex2']:
    PSF = io.imread(os.path.join('outputs', 'figures', dataset_name,\
        'scale_1_gauss_0_poiss_0_ratio_1', 'kernels_bc_1_re_1',\
        'kernel_fp.tif')).astype(np.float32)

PSF = np.transpose(PSF, axes=(2, 0, 1))

# ------------------------------------------------------------------------------
# evaluation metrics
cal_ssim = lambda x, y: eva.SSIM(img_true=y, img_test=x,\
    data_range=y.max() - y.min(), version_wang=False, channle_axis=0)
cal_mse  = lambda x, y: eva.PSNR(img_true=y, img_test=x,\
    data_range=y.max() - y.min())
cal_ncc  = lambda x, y: eva.NCC(img_true=y, img_test=x)
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

print('-'*80)
# ------------------------------------------------------------------------------
for id in id_data:
    data_gt = io.imread(os.path.join(data_gt_path, test_txt[id]))\
        .astype(np.float32)
    data_input = io.imread(os.path.join(data_raw_path, test_txt[id]))\
        .astype(np.float32)

    print('Load data from:', data_raw_path)
    print('GT: {}, input: {}, PSF: {}'\
        .format(list(data_gt.shape), list(data_input.shape), list(PSF.shape)))

    # --------------------------------------------------------------------------
    # save result to path
    if dataset_name in ['Microtubule', 'Nuclear_Pore_complex']:
        fig_path = os.path.join('outputs', 'figures', dataset_name.lower(),\
            'scale_1_noise_0_ratio_1', f'sample_{id}')

    if dataset_name in ['Microtubule2', 'Nuclear_Pore_complex2']:
        fig_path = os.path.join('outputs', 'figures', dataset_name.lower(),\
            'scale_1_gauss_0_poiss_0_ratio_1', f'sample_{id}')

    if not os.path.exists(fig_path): os.makedirs(fig_path)
    print('Save results to :', fig_path)

    # for meth in ['traditional','gaussian','butterworth','wiener_butterworth']:
    for meth in ['traditional']:
        meth_path = os.path.join(fig_path, meth)
        if not os.path.exists(meth_path): os.makedirs(meth_path)

    # --------------------------------------------------------------------------
    if enable_traditonal:
        out_trad = DCV_trad.deconv(data_input, num_iter=num_iter_trad,\
            domain='direct')
        bp_trad  = DCV_trad.PSF2
        out_gaus_metrics = DCV_trad.get_metrics() 

        io.imsave(fname=os.path.join(fig_path, 'traditional',\
            f'deconv_{num_iter_trad}.tif'), arr=out_trad, check_contrast=False)
        io.imsave(fname=os.path.join(fig_path, 'traditional', 'deconv_bp.tif'),\
            arr=bp_trad, check_contrast=False)
    print('-'*80)
    print(tabulate(out_gaus_metrics.transpose()))
    print('-'*80)

    # # ------------------------------------------------------------------------
    # if enable_gaussian:
    #     out_gaus    = DCV_gaus.deconv(data_input, num_iter=num_iter_gaus,\
    #         domain='fft')
    #     bp_gaus     = DCV_gaus.PSF2
    #     bp_gaus_otf = DCV_gaus.OTF_bp
    #     out_gaus_metrics = DCV_gaus.get_metrics()

    #     io.imsave(fname=os.path.join(fig_path, 'gaussian', 'deconv.tif'),\
    #         arr=out_gaus, check_contrast=False)
    #     io.imsave(fname=os.path.join(fig_path, 'gaussian', 'deconv_bp.tif'),\
    #         arr=bp_gaus, check_contrast=False)

    # # ------------------------------------------------------------------------
    # if enable_bw:
    #     out_bw    = DCV_butt.deconv(data_input, num_iter=num_iter_bw,\
    #         domain='fft')
    #     bp_bw     = DCV_butt.PSF2
    #     bp_bw_otf = DCV_butt.OTF_bp
    #     out_bw_metrics = DCV_butt.get_metrics()

    #     io.imsave(fname=os.path.join(fig_path,'butterworth','deconv.tif'),\
    #         arr=out_bw, check_contrast=False)
    #     io.imsave(fname=os.path.join(fig_path,'butterworth','deconv_bp.tif'),\
    #         arr=bp_bw, check_contrast=False)

    # # ------------------------------------------------------------------------
    # if enable_wb:
    #     out_wb    = DCV_wb.deconv(data_input, num_iter=num_iter_wb,\
    #         domain='fft')
    #     bp_wb     = DCV_wb.PSF2
    #     bp_wb_otf = DCV_wb.OTF_bp
    #     out_wb_metrics = DCV_wb.get_metrics()

    #     io.imsave(fname=os.path.join(fig_path, 'wiener_butterworth',\
    #         'deconv.tif'), arr=out_wb, check_contrast=False)
    #     io.imsave(fname=os.path.join(fig_path, 'wiener_butterworth',\
    #         'deconv_bp.tif'), arr=bp_wb, check_contrast=False)
    # # ------------------------------------------------------------------------
