# Kernel Learning Deconvolution (KLD)

These are the source python codes and instructions for KLD.

This package includes:
- MATLAB implementation of training data simulation
- Python implementation of training and inference of KLD
- Python implementation of conventional Richardson-Lucy Deconvolution (RLD) with different backward kernels, includeing traditional, Gaussian, Butterworth, Wiener-Butterworth (WB).

## File structure
`./checkpoints`: include the saved models for different data set.

`./data`: includes the example simulation data used to training and test methods, and partial publicly accessiable fluorescence microscopic images (including BioSR, Confocal/STED volumes, LLSM volumes). The complete data sets are accessible from the original repositories.

`./methods`: includes the MATLAB codes of `DeconvBlind` and Python codes of RLD using different backward kernsls.

`./models`: includes the Python codes of `KLD` (`./models/kernelnet.py`). 

`./outputs`: saves the output results of `evaluate_model.py`.

`./Phantom_generate`: includes the MATLAB codes used to generate simulation phantoms.

`./utils`: includes the functions used to processing data, quantitatively evaluation, plot images.


## Enviroment
We run our codes on Windows 11 (optional) with CPU. The version of Python is 3.11.9, which must higher then 3.7.

The python package used in our projects:
- torch==2.0
- torchvision
- tensorboard
- numpy
- matplotlib
- scikit-image
- pydicom
- pytorch-msssim
- fft-conv-pytorch

To use our code, you shold create a virtual enviroment and install the required packages first:

```
$ conda create -n pytorch python=3.11.9 
$ conda activate pytorch
$ pip install -r requirements.txt
```
The cd into the python folder

```
cd .\python\
```

**Please always pay attention to the path setting in the code, and modifiy the to your own working path.**
**Specific parameters can be modified according to you needs.**

## Training data set generation
### Simulation data set
1. We use MATLAB code in [Richardson-Lucy-Net](https://github.com/MeatyPlus/Richardson-Lucy-Net/tree/main/Phantom_generate) to generate simulated phantoms with bead structures or mixed structures. The modified codes are save in `Phantom_generate` folder.(please modify the save path to `data`)

2. Then, we use `generate_synthetic_data.py` to generate the simulated data sets with different noise level. The generated data sets will be saved in `./data/RLN/`

### Real biology images
For the images in BioSR and Confocal/STED volume data set, the images should be preprocessing before training. 
We use `real_data_preprocessing.py` to preprocess the real biology images in Confocal/STED volume data set.
The preprocessing file `preprocess.py` save in `data/BioSR` is used to proprocess the data of BioSR. 

## Train a new model
We use `main_kernelnet.py` to train a new model. (please modify the `root_path` to the path saved data sets) The training parameters can be modified directly in the source code.

The model weights will be saved in `./checkpoints` folder.

## Test a trained model
To test a well-trained model, we use `evaluate_model.py`, the output results will be save in `./outputs/figures`.

Some pre-trained models are provides in `./checkpoints`.

## Other files
- `deconv3D_w_gt.py` is used to deconv the 3D images in `simulation` data set, using conventional RLD methods.

- `deconv3D_live.py` is used to deconv the 3D volumes in `LLSM volume` data set using conventional RLD methods. The parameter `id_sample` and `wave_length` should be modified according to your data directory. Please enable specific method to do deconvolution.

- `deconv2D_real.py` is used to deconv the real 2D biological images in `BioSR` data set using conventional RLD methods. As there is no PSF is provided, the PSF are learned form the paired data, and then used in this file. You should use `main_kernelnet.py` to train the model, and use `evaluate_model.py` to generate the learned PSF, and then use it to do deconvolution.

- `deconv3D_real.py` is used to deconv the real 3D biological images in `Confocal/STED volume` data set using conventional RLD methods. As there is no PSF is provided, the PSF are learned form the paired data, and then used in this file. You should use `main_kernelnet.py` to train the model, and use `evaluate_model.py` to generate the learned PSF, and then use it to do deconvolution.















