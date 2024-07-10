# @ https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py#L213

import numpy as np
import skimage.io as io
import os

def gaussian_kernel(n=3, std=1.0):
    '''
    Geenrate 2D gaussian kernel.
    - n (int), kernel size.
    - std (float), standard deviation.
    '''
    x, y = np.mgrid[-(n//2):n//2 + 1, -(n//2):n//2 + 1]
    hg = np.exp(-(x**2 + y**2) / (2 * (std**2)))
    h  = hg / np.sum(hg)
    return h

def psf2otf(psf, outSize=None):
    '''
    Convert point-spread funciton to optical transfer function.
    Computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering. By
    default, the OTF array is the same size as the PSF array.
    Args:
    - psf (numpy.ndarray): PSF array.
    '''
    if np.all(psf == 0): return np.zeros_like(psf)

    psfSize = np.asarray(psf.shape, dtype=np.int8)
    outSize = np.asarray(outSize, dtype=np.int8)
    padSize = outSize - psfSize

    # Pad the PSF to outsize
    if len(psf.shape) == 2:
        psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    if len(psf.shape) == 3:
        psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1]), (0, padSize[2])), 'constant')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(psfSize):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the conputation of the FFT.
    nElem = np.prod(psfSize, axis=0)
    nOps = 0
    for k in range(0, np.ndim(psf)):
        nffts = nElem / psfSize[k]
        nOps  = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts

    # Discard the imaginary part of the psf if it's within roundoff error.
    mx1 = (abs(np.imag(otf[:])).max(0)).max(0)
    mx2 = (abs(otf[:]).max(0)).max(0)
    eps = 2.2204e-16
    if mx1 / mx2 <= nOps * eps: otf = np.real(otf)
    return otf

def corelucy(Y,H,DAMPAR22,wI,READOUT,SUBSMPL,idx,vec,num):
    '''
    Accelerated Damped Lucy-Richardson Operator.
    Calculates fucntion that when used witht he scaled projected array
    produces the next iteration array that maximizes the likelihood that
    the entire suite satisfies the Possion statistics.
    ''' 
    ReBlurred = np.real(np.fft.ifftn(H * np.fft.fftn(Y)))
    # Resampling if needed
    if SUBSMPL != 1:
        # Bin ReBlurred back to the sizeI for non-singleton dims

        # Reshape so that the-to-binned dimension separates into two
        # dimensions, with one of them consisting of elements of a single bin.
        ReBlurred = np.reshape(ReBlurred, vec)

        # Bin (==calculate mean) along the first of the-to-binned dimension,
        # that dimension consists of the bin elements. Reshape to get rid off


        # An estimate for the next step
        eps = 2.2204e-16
        ReBlurred = ReBlurred + READOUT
        ReBlurred = np.where(ReBlurred == 0, eps, ReBlurred)
        AnEstim = wI / ReBlurred + eps

        # Dampling if needed
        if DAMPAR22 == 0:
            pass
        else:
            # Dampling of the image relative the DAMPAR22 = (N*sigma)^2
            gm = 10
            g = (wI * np.log(AnEstim)+ ReBlurred - wI) / DAMPAR22
            g = np.min(g, 1)
            G = (g^(gm - 1)) * (gm - (gm - 1) * g)
            ImRatio = 1 + G * (AnEstim[idx{:}] - 1)

        f = np.fft.fftn(ImRatio)



def deconvlucy(I, PSF, NUMIT=10, DAMPAR=0, WEIGHT=None, READOUT=0, SUBSMPL=1.0):
    '''
    Args:
    - I (numpy.ndarray): the input array (could be 2D, 3D)
    - PSF:      operator that distorts the ideal image.
    - NUMIT (int): Number of oterations, usually produces good result by 10.
    - DAMPAR:   No dampling is default.
    - WEIGHT: Assigned to each pixel to reflect its recoding quality in the camera.
    - READOUT:  Zero readout noise or any other back/fore/fround noise associated with CCD camera.
                Or the image is corrected already for this noise by user.
    - SUBSMPL: Image and PSF are given at equal resolution, no over/under sampling at all.
    '''
    sizeI    = np.asarray(I.shape)
    sizePSF  = np.asarray(PSF.shape)
    numNSdim = np.nonzero(sizePSF!=1)[0]
    J = {}
    J[1] = I
    assert np.max(J[1].shape) >=2,\
        print('deconvolucy: input images must have at least 2 elements.')

    Jlength = len(J)
    if Jlength == 1:
        J[2] = J[1]
        J[3] = 0
    if len(J) == 3:
        J[4] = 0

    # J{1}=I, J{2} is the image resulted from
    # the last iteration, J{3} is the image from one before last iteration,
    # J{4} is an array used internally by the iterative algorithm.
    


    assert np.prod(sizePSF) >= 2,\
        print('deconvolucy: psf must have at least 2 elements.')
    assert np.all(PSF[:] == 0) == False,\
        print('deconvlucy: psf must not be zero everywhere.')

    if WEIGHT == None: 
        WEIGHT = np.ones(sizeI)

    # Prepare PSF.If PSF is known at a higher sampling rate, it has to be
    # padded with zeros up to sizeI(numNSdim)*SUBSMPL in all non-singleton
    # dimensions. Or its OTF could take care of it:
    sizeOTF = sizeI
    for ind in numNSdim.tolist(): 
        sizeOTF[ind] = SUBSMPL * sizeI[ind]
    H = psf2otf(PSF, sizeOTF)

    # Prepare parameter for iterations
    # create indexes for image according to the sampling rate


    # L_R Iterations
    lam = 0
    for k in range(NUMIT):
        # Make an image prediction for the next iteration
        if k > 2:
            lam = 0

        # Make core for the LR estimation
        CC = corelucy(Y, H, DAMPAR22, wI, READOUT, SUBSMPL, idx, vec, num)

        # Determine next iteration image and apply positivity constraint

        # Convert the right array to the original image class and output whole thing


if __name__ == '__main__':
    psf = gaussian_kernel(3, std=1.0)
    otf = psf2otf(psf, (5,5))
    img_path = os.path.join('data','lena','Lena.tif')
    img_gt = io.imread(img_path) / 255.0

    deconvlucy(I=img_gt, PSF=psf)

    print('end')