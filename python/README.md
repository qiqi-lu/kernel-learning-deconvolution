# Kernel Learning Deconvolution (KLD)

These are the source python codes and instructions for KLD.

This package includes:
- Python implementation of training and inference of KLD
- MATLAB implementation of training data simulation
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
We run our codes on Windows 11 (optional) without GPU. The version of Python is 3.11.9, which must high then 3.7.

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
$ conda create -n kld python=3.11.9 
$ conda activate kld
$ pip install -r requirements.txt
```
Please always pay attention to the path setting in the code, and modifiy the to your own working path.

## Training data set generation
### Simulation data set
1. We use MATLAB code in [Richardson-Lucy-Net](https://github.com/MeatyPlus/Richardson-Lucy-Net/tree/main/Phantom_generate) to generate simulated phantoms with bead structures or mixed structures. The modified codes are save in `Phantom_generate` folder.(please modify the save path to `data`)

2. We use `generate_synthetic_data.py` to generate the simulated data sets with different noise level. The generated data sets will be saved in `./data/RLN/`

### Real biology images
For the images in BioSR and Confocal/STED volume data set, the images should be preprocessing before training. 
We use `real_data_preprocessing.py` to preprocess the real biology images in BioSR and Confocal/STED volume data set.

## Train a new model
We use `main_kernelnet.py` to train a new model. (please modify the `root_path` to the path saved data sets)

The model weights will be saved in `./checkpoints` folder.

## Test a trained model
To test a well-trained model, we use `evaluate_model.py`, the output results will be save in `./outputs/figures`.

Some pre-trained models are provides in `./checkpoints`.

## Other files
The codes in `deconv2D_real.py`, `deconv3D_real.py`, `deconv3D_live.py`, and `deconv3D_w_gt.py` are used to deconv the real 2D biological images in `BioSR` data set, real 3D biological images in `Confocal/STED volume` data set, 3D volumes in `LLSM volume` data set, 3D images in `simulation` data set, respectively, using conventional RLD methods.













