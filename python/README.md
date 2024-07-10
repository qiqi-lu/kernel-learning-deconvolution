# KLD (Kernel Learning Deconvolution)

This repsitory consists of the python codes for the paper "**Kernel learning enables fluorescence microscopic image deconvolution with enhanced performance and speed**"

## Introduction 
KLD is an algorithm for fluorescence microscopic image deconvolution.
It enhances the deconvolution performance and speed through learning the forward kernel and backward kernel in conventional Richardson-Lucy Deconvolution (RLD) algorithm.

KLD only requires one training sample and two iterations to achieve superior deconvolution perfromance and speed compared to traditional RLD and its variants (which using an unmatched backeard kernel, such as Gaussian, Butterworth, and Wiener-Butterworth (WB) backward kernels).

