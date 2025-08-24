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
The first exercise, [1. Context-aware restoration (CARE)](01_CARE/exercise.ipynb), is mandatory. It will introduce you to deep learning image restoration by training a UNet for denoising.

When you reach the end of the first exercise, choose your next exercise from the following:

[2. Noise2Void (N2V)](02_Noise2Void/exercise.ipynb).
This exercise will introduce a denoiser that is trained using only noisy data.
Unlike CARE, it doesn't need any examples of clean images.
It's also relatively quick to train.
But there's a catch.
It relies on the assumption that the noise is unstructured.
Unstructured noise is uncorrelated over pixels, so has no streaky or line artifacts.
An example is shown below.

<img src="./02_Noise2Void/imgs/unstructured noise.png">

[3. COSDD](03_COSDD/exercise.ipynb). 
This exercise will introduce another denoiser that is trained using unpaired noisy images, but can handle a specific form of structure.
That structure is row correlation.
Row-correlated noise is common in scanning-based imaging techniques like point-scanning confocal microscopy.
It can also be found when using sCMOS sensors.
The practical trade-off with N2V is that COSDD takes much longer to train.
An example of row-correlated noise is shown below.

<img src="./03_COSDD/resources/structured noise.png">

[4. MicroSplit](04_MicroSplit/exercise.ipynb).
This exercise introduces a method that not only learns to denoise images, but also performs computational unmixing.
Specifically, it uses deep learning to separate multiple superimposed cellular structures within a single fluorescent image channel, turning one fluorescent channel into as many as four.
Imaging multiple cellular structures in a single fluorescent channel effectively increases the available photon budget, which can be reallocated to achieve faster imaging, higher signal-to-noise ratios, or the imaging of additional structures. 
An example of splitting is shown below.

<img src="./04_MicroSplit/imgs/Fig1_b.png">

## Bonus exercise
If you've finished these exercises, have a look at [5. Noise2Noise](05_bonus_Noise2Noise).
This is a denoiser that is similar to CARE, except instead of noisy inputs and clean targets it uses both noisy inputs and noisy targets, i.e. two images with the same underlying signal but different realisations of noise.
Because the random noise content of an image is on average zero, a network trained with noisy targets will converge to the same optima as a network trained with clean targets.

