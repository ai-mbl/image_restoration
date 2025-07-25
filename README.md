# Image Restoration: denoising and splitting


Welcome to the Image Restoration exercises. In this part of the course, we will explore
how to use deep learning to denoise images, with examples of widely used algorithm for
both supervised and unsupervised denoising. We will also explore the difference
between unstructured and structured noise, and between UNet (which you are familiar with
by now) and VAE architectures (see `COSDD` exercise)!
We'll also tackle the task of image splitting (or unmixing) where a single image exhibiting superimposed labeled structures is decomposed in multiple channels, each one corresponding to a different structure using the `MicroSplit` algorithm.


## Setup

Please run the setup script to create the environment for these exercises and download data.

``` bash
source setup.sh
```

## Exercises

1. [Context-aware restoration](01_CARE/care_exercise.ipynb)
2. [Noise2Void](02_Noise2Void/n2v_exercise.ipynb)
3. [Correlated and Signal Dependent Denoising (COSDD)](03_COSDD/exercise.ipynb)
4. [MicroSplit](04_MicroSplit/exercise.ipynb)


## Bonus

- [Noise2Noise](05_bonus_Noise2Noise/n2n_exercise.ipynb)


