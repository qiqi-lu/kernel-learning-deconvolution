'''
Use conventional deconvolution method to restore 3D image.
Requirements:
- PSF
'''

import numpy as np
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.dataset_utils as utils_data

# ------------------------------------------------------------------------------
dataset_name = 'ZeroShotDeconvNet'
# id_sample = [0, 346]
# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = list(range(1, 500, 10))
id_sample = range(0, 1000, 4)
# id_sample = [1]

# wave_length = '642'
wave_length = '560'

enable_traditonal, enable_gaussian, enable_bw, enable_wb = 0, 0, 0, 1
num_iter_trad, num_iter_gaus, num_iter_bw, num_iter_wb = 30, 30, 30, 2 

if wave_length == '642':
    dataset_path = os.path.join('F:', os.sep, 'Datasets', dataset_name,\
        '3D time-lapsing data_LLSM_Mitosis_H2B', '642')
        
if wave_length == '560':
    dataset_path = os.path.join('F:', os.sep, 'Datasets', dataset_name,\
        '3D time-lapsing data_LLSM_Mitosis_Mito', '560')

subfolder = os.path.join('Mitosis', wave_length)

# ------------------------------------------------------------------------------
data_raw_path = os.path.join(dataset_path, 'raw')
with open(os.path.join(dataset_path, 'raw.txt')) as f:
    test_txt = f.read().splitlines() 

# ------------------------------------------------------------------------------
print('Load PSF ...')
PSF_raw = io.imread(os.path.join(data_raw_path, 'PSF.tif')).astype(np.float32)
print('PSF shape (raw): {} (sum = {})'.format(PSF_raw.shape, PSF_raw.sum()))
PSF_raw = PSF_raw / PSF_raw.sum()

# even shape to odd shape
if PSF_raw.shape[-1] % 2 == 0:
    print('Convert PSF with even shape to odd shape.')
    PSF_odd = utils_data.even2odd(PSF_raw)
    PSF = utils_data.center_crop(PSF_odd, size=(101, 101, 101))
    print('PSF shape: {} (sum = {:>.4f})'.format(PSF.shape, PSF.sum()))
    PSF = PSF / PSF.sum()
    save_to = os.path.join(data_raw_path,'PSF_odd.tif')
    io.imsave(fname=save_to, arr=PSF, check_contrast=False)
    print('Save PSF to', save_to)
else:
    PSF = PSF_raw

# ------------------------------------------------------------------------------
# DCV_trad = dcv.Deconvolution(PSF=PSF, bp_type='traditional', init='measured')
# DCV_gaus = dcv.Deconvolution(PSF=PSF, bp_type='gaussian',    init='measured')
# DCV_butt = dcv.Deconvolution(PSF=PSF, bp_type='butterworth', beta=0.01, n=10,\
#     res_flag=1, init='measured')
DCV_wb = dcv.Deconvolution(PSF=PSF, bp_type='wiener-butterworth', alpha=0.005,\
    beta=0.1, n=10, res_flag=1, init='measured')

# ------------------------------------------------------------------------------
for id in id_sample:
    load_from = os.path.join(data_raw_path, test_txt[id])
    data_raw = io.imread(load_from).astype(np.float32)
    print('Load data from: ', load_from)
    print('Input shape: {}'.format(list(data_raw.shape)))

    # save result to path
    fig_path = os.path.join('outputs', 'figures', dataset_name.lower(),\
        f'{subfolder}', f'sample_{id}')

    print('Save results to :', fig_path)

    # for meth in ['traditional','gaussian','butterworth','wiener_butterworth']:
    for meth in ['wiener_butterworth']:
        meth_path = os.path.join(fig_path, meth)
        if not os.path.exists(meth_path): os.makedirs(meth_path)

    # # --------------------------------------------------------------------------
    # if enable_traditonal:
    #     out_trad = DCV_trad.deconv(data_raw, num_iter=num_iter_trad,\
    #         domain='fft')
    #     bp_trad = DCV_trad.PSF2

    #     io.imsave(fname=os.path.join(fig_path, 'traditional', 'deconv.tif'),\
    #         arr=out_trad, check_contrast=False)
    #     if id == 0: 
    #         io.imsave(fname=os.path.join(fig_path, 'traditional',\
    #             'deconv_bp.tif'), arr=bp_trad, check_contrast=False)

    # # --------------------------------------------------------------------------
    # if enable_gaussian:
    #     out_gaus = DCV_gaus.deconv(data_raw, num_iter=num_iter_gaus,\
    #         domain='fft')
    #     bp_gaus = DCV_gaus.PSF2

    #     io.imsave(fname=os.path.join(fig_path, 'gaussian', 'deconv.tif'),\
    #         arr=out_gaus, check_contrast=False)
    #     if id == 0: 
    #         io.imsave(fname=os.path.join(fig_path, 'gaussian',\
    #             'deconv_bp.tif'), arr=bp_gaus, check_contrast=False)

    # # --------------------------------------------------------------------------
    # if enable_bw:
    #     out_bw = DCV_butt.deconv(data_raw, num_iter=num_iter_bw, domain='fft')
    #     bp_bw  = DCV_butt.PSF2

    #     io.imsave(fname=os.path.join(fig_path, 'butterworth', 'deconv.tif'),\
    #         arr=out_bw, check_contrast=False)
    #     if id == 0: 
    #         io.imsave(fname=os.path.join(fig_path, 'butterworth',\
    #             'deconv_bp.tif'), arr=bp_bw, check_contrast=False)

    # # --------------------------------------------------------------------------
    if enable_wb:
        out_wb = DCV_wb.deconv(data_raw, num_iter=num_iter_wb, domain='fft')
        bp_wb  = DCV_wb.PSF2

        io.imsave(fname=os.path.join(fig_path, 'wiener_butterworth',\
            'deconv.tif'), arr=out_wb.astype(np.uint16), check_contrast=False)
        if id == 0:
            io.imsave(fname=os.path.join(fig_path, 'wiener_butterworth',\
                'deconv_bp.tif'), arr=bp_wb, check_contrast=False)