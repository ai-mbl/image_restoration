# Image Restoration: denoising and splitting


Welcome to the Image Restoration exercises. In this part of the course, we will explore
how to use deep learning to denoise images, with examples of widely used algorithm for
both supervised and unsupervised denoising. We will also explore the difference
between unstructured and structured noise, or between UNet (which you are familiar with
by now) and VAE architectures (see COSDD exercise)!

Finally, we have bonus exercises for those wanted to explore more denoising algorithms or
image splitting!


## Setup

Please run the setup script to create the environment for these exercises and download data.

``` bash
source setup.sh
```


When you are ready to start the exercise, make sure you are in the `05_image_restoration` environment and then run jupyter lab.

``` bash
conda activate 05_image_restoration
jupyter lab
```

## Exercises

1. [Context-aware restoration](01_CARE/care_exercise.ipynb)
2. [Noise2Void](02_Noise2Void/n2v_exercise.ipynb)
3. [COSDD](03_COSDD/exercise.ipynb)
4. [DenoiSplit](04_DenoiSplit/denoisplit.ipynb)


## Bonus

- [Noise2Noise](04_bonus_Noise2Noise/n2n.ipynb)


