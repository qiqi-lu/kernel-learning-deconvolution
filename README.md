# Kernel Learning Deconvolution (KLD)

This repsitory consists of the python codes for the paper "**Kernel learning enables fluorescence microscopic image deconvolution with enhanced performance and speed**"

## Introduction 
KLD is an algorithm for fluorescence microscopic image deconvolution.
It enhances the deconvolution performance and speed through learning the forward kernel and backward kernel in conventional Richardson-Lucy Deconvolution (RLD) algorithm.

KLD only requires one training sample and two iterations to achieve superior deconvolution perfromance and speed compared to traditional RLD and its variants (which using an unmatched backeard kernel, such as Gaussian, Butterworth, and Wiener-Butterworth (WB) backward kernels).

## Instructions
The source codes (Python and MATLAB) are saved in `python`.

**For derails instructions, see the `README.md` in the `python` folder**

## napari-kld plugin
A `napari` plugin, named `napari-kld`, was developed for KLD.

The source code is saved at https://github.com/qiqi-lu/napari-kld, please go to this repository for more information.

You can install `napari-kld` via [pip](https://pypi.org/project/pip/):

```
pip install napari-kld
```

And it is also accessable through `napari hub` at https://www.napari-hub.org/plugins/napari-kld.

*You should install `napari` for using `napari-kld` plugin.*



