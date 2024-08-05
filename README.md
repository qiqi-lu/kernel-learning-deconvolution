# Kernel Learning Deconvolution (KLD)

This repository consists of the python codes for the paper "**Kernel learning enables fluorescence microscopic image deconvolution with enhanced performance and speed**"

## Introduction 
KLD is an algorithm for fluorescence microscopic image deconvolution.
It enhances the deconvolution performance and speed through learning the forward kernel and backward kernel in conventional Richardson-Lucy Deconvolution (RLD) algorithm.

KLD only requires one training sample and two iterations to achieve superior deconvolution perfromance and speed compared to traditional RLD and its variants (which using an unmatched backeard kernel, such as Gaussian, Butterworth, and Wiener-Butterworth (WB) backward kernels).

## Instruction
The source codes (Python and MATLAB) are saved in `"python"` folder. We have developed a `napari` plugin for KLD, named `napari-kld`. And, some data used for testing `napari-kld` plugin are saved in `"test"` folder.

**For derails instructions of source Python codes, see the `"README.md"` in the `"python"` folder.**

**For the instructions of `napari-kld` plugin, please see next section.**

## napari-kld plugin
`mapari-kld` is a `napari` plugin, named `napari-kld`, was developed for KLD.

The source code is saved at https://github.com/qiqi-lu/napari-kld, please go to this repository for more information. And it is also accessable through `napari hub` at https://www.napari-hub.org/plugins/napari-kld.

### Installation

You should install `napari` firstly and then install `napari-kld`.

#### **Install `napari`**

You can download the `napari` bundled app for a simple installation via https://napari.org/stable/tutorials/fundamentals/quick_start.html#installation.

Or, you can install `napari` with Python using pip:

```
conda create -y -n napari-env -c conda-forge python=3.10
conda activate napari-env
python -m pip install 'napari[all]'
```

Refer to https://napari.org/stable/tutorials/fundamentals/quick_start.html#installation.

#### **Install `napari-kld`**

You can install napari-kld plugin with napari:

Plugins > Install/Uninstall Plugins… > [input napari-kld] > install

You can install `napari-kld` via [pip](https://pypi.org/project/pip/):

```
pip install napari-kld
```
