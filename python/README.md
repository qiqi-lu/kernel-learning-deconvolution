# KLD (Kernel Learning Deconvolution)

These are the source python codes and instructions for KLD.

This package includes:
- Python implementation of training and inference of KLD
- MATLAB implementation of training data simulation
- Python implementation of conventional Richardson-Lucy Deconvolution (RLD) with different backward kernels, includeing traditional, Gaussian, Butterworth, Wiener-Butterworth (WB).

## File structure
`data stes`: includes the example simulation data used to training and test methods.

`models`: includes the Python codes of `KLD`.

`methods`: includes the Matlab codes of `DeconvBlind` and Python codes of RLD using different backward kernsls.

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
## Training data set generation
### Simulation data set
We use MATLAB code in [Richardson-Lucy-Net](https://github.com/MeatyPlus/Richardson-Lucy-Net/tree/main/Phantom_generate) to generate simulated phantoms with bead structures or mixed structures. The modified codes are save in `Phantom_generate` folder.(please modify the data save path to `data sets`)








