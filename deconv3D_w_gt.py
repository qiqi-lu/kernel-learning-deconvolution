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

# ------------------------------------------------------------------------------
dataset_name, psf_size = 'SimuBeads3D_128', 31
# dataset_name, psf_size = 'SimuMix3D_128', 31
# dataset_name, psf_size = 'SimuMix3D_256', 31
# dataset_name, psf_size = 'SimuMix3D_382', 127
# ------------------------------------------------------------------------------
std_gauss, poisson, ratio = 0, 0, 1
# std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 0.1
# ------------------------------------------------------------------------------
id_data = [0, 1, 2, 3, 4, 5, 6]
# id_data = [2]

scale_factor = 1
domain = 'fft'

# ------------------------------------------------------------------------------
enable_traditonal, enable_gaussian, enable_bw, enable_wb = 0, 0, 0, 1
# num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 100, 100, 100, 30 
num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 2, 2, 2, 2

# ------------------------------------------------------------------------------
# load data
dataset_path  = os.path.join('F:', os.sep, 'Datasets', 'RLN', dataset_name)
data_gt_path  = os.path.join(dataset_path, 'gt')
data_raw_path = os.path.join(dataset_path,\
    'raw_psf_{}_gauss_{}_poiss_{}_sf_{}_ratio_{}'\
        .format(psf_size, std_gauss, poisson, scale_factor, ratio))

with open(os.path.join(dataset_path, 'test.txt')) as f:
    test_txt = f.read().splitlines() 

PSF = io.imread(os.path.join(data_raw_path, 'PSF.tif')).astype(np.float32)

# ------------------------------------------------------------------------------
# evaluation metrics
cal_ssim = lambda x, y: eva.SSIM(img_true=y, img_test=x,\
    data_range=y.max() - y.min(), multichannel=False, channle_axis=None,\
    version_wang=False)
cal_mse  = lambda x, y: eva.PSNR(img_true=y, img_test=x,\
    data_range=y.max() - y.min())
cal_ncc  = lambda x, y: eva.NCC(img_true=y, img_test=x)
metrics  = lambda x : [cal_mse(x, data_gt), cal_ssim(x, data_gt),\
    cal_ncc(x, data_gt)]

# ------------------------------------------------------------------------------
DCV_trad = dcv.Deconvolution(PSF=PSF, bp_type='traditional', init='measured',\
    metrics=metrics)
DCV_gaus = dcv.Deconvolution(PSF=PSF, bp_type='gaussian', init='measured',\
    metrics=metrics)
DCV_butt = dcv.Deconvolution(PSF=PSF, bp_type='butterworth', beta=0.01, n=10,\
    res_flag=1, init='measured', metrics=metrics)

if ratio in [1, 0.3]:
    DCV_wb = dcv.Deconvolution(PSF=PSF, bp_type='wiener-butterworth',\
        alpha=0.005, beta=0.1, n=10, res_flag=1, init='measured',\
        metrics=metrics) # default
if ratio == 0.1:
    DCV_wb = dcv.Deconvolution(PSF=PSF, bp_type='wiener-butterworth',\
        alpha=0.005, beta=0.001, n=10, res_flag=1, init='measured',\
        metrics=metrics)

# ------------------------------------------------------------------------------
for id in id_data:
    data_gt = io.imread(os.path.join(data_gt_path, test_txt[id]))
    data_gt = data_gt * ratio
    data_input = io.imread(os.path.join(data_raw_path, test_txt[id]))

    data_gt = data_gt.astype(np.float32)
    data_input = data_input.astype(np.float32)

    print('>> Load data from: ', data_raw_path)
    print('>> GT: {}, Input: {}, PSF: {}'\
        .format(data_gt.shape, data_input.shape, PSF.shape))

    # --------------------------------------------------------------------------
    # save result to path
    path_fig = os.path.join('outputs', 'figures', dataset_name.lower(),\
        'scale_{}_gauss_{}_poiss_{}_ratio_{}'\
        .format(scale_factor, std_gauss, poisson, ratio),\
        f'sample_{id}')
    if not os.path.exists(path_fig): os.makedirs(path_fig)

    print('Save results to :', path_fig)

    for meth in ['traditional','gaussian','butterworth','wiener_butterworth']:
        path_meth = os.path.join(path_fig, meth)
        if not os.path.exists(path_meth): os.makedirs(path_meth)

    if enable_traditonal:
        out_trad = DCV_trad.deconv(data_input, num_iter=num_iter_trad,\
            domain=domain)
        bp_trad  = DCV_trad.PSF2
        # bp_trad_otf = DCV_trad.OTF_bp
        out_trad_metrics = DCV_trad.get_metrics()

        io.imsave(fname=os.path.join(path_fig, 'traditional',\
            f'deconv_{num_iter_trad}.tif'), arr=out_trad, check_contrast=False)
        io.imsave(fname=os.path.join(path_fig, 'traditional',\
            'deconv_bp.tif'), arr=bp_trad, check_contrast=False)
        np.save(file=os.path.join(path_fig,    'traditional',\
            f'deconv_metrics_{num_iter_trad}.npy'), arr=out_trad_metrics)

    # ------------------------------------------------------------------------------
    if enable_gaussian:
        out_gaus = DCV_gaus.deconv(data_input, num_iter=num_iter_gaus, domain=domain)
        bp_gaus  = DCV_gaus.PSF2
        out_gaus_metrics = DCV_gaus.get_metrics()

        io.imsave(fname=os.path.join(path_fig, 'gaussian',\
            f'deconv_{num_iter_gaus}.tif'),  arr=out_gaus, check_contrast=False)
        io.imsave(fname=os.path.join(path_fig, 'gaussian',\
            'deconv_bp.tif'), arr=bp_gaus, check_contrast=False)
        np.save(file=os.path.join(path_fig,    'gaussian',\
            f'deconv_metrics_{num_iter_gaus}.npy'), arr=out_gaus_metrics)

    # ------------------------------------------------------------------------------
    if enable_bw:
        out_bw = DCV_butt.deconv(data_input, num_iter=num_iter_bw, domain=domain)
        bp_bw  = DCV_butt.PSF2
        out_bw_metrics = DCV_butt.get_metrics()

        io.imsave(fname=os.path.join(path_fig, 'butterworth',\
            f'deconv_{num_iter_bw}.tif'), arr=out_bw, check_contrast=False)
        io.imsave(fname=os.path.join(path_fig, 'butterworth',\
            'deconv_bp.tif'), arr=bp_bw, check_contrast=False)
        np.save(file=os.path.join(path_fig,    'butterworth',\
            f'deconv_metrics_{num_iter_bw}.npy'), arr=out_bw_metrics)

    # ------------------------------------------------------------------------------
    if enable_wb:
        out_wb = DCV_wb.deconv(data_input, num_iter=num_iter_wb, domain=domain)
        bp_wb  = DCV_wb.PSF2
        out_wb_metrics = DCV_wb.get_metrics()

        io.imsave(fname=os.path.join(path_fig, 'wiener_butterworth',\
            f'deconv_{num_iter_wb}.tif'), arr=out_wb, check_contrast=False)
        io.imsave(fname=os.path.join(path_fig, 'wiener_butterworth',\
            'deconv_bp.tif'), arr=bp_wb, check_contrast=False)
        np.save(file=os.path.join(path_fig,    'wiener_butterworth',\
            f'deconv_metrics_{num_iter_wb}.npy'), arr=out_wb_metrics)