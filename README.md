# Exercise 3: Image Restoration

## Setup

Conda environments are useful to keep your workflows running, even if they have conflicting dependencies. For examples,
one method might be using TensorFlow 1.x and the old python 2.7, while another uses TensorFlow 2.x and some python 3
version.

This exercise is split into several parts. They use different libraries so to keep things tidy we will set up separate conda environments

For the first and second part, we will set up a conda environment for CARE/CSBDeep. Make sure to first deactivate a previous conda environment, e.g. from the last exercise.

```
# create a new conda environment called 'care' and initialize it with python version 3.7
conda create -n care python=3.7
# activate the environment
conda activate care
# install necessary dependencies
conda install tensorflow-gpu keras jupyter tensorboard nb_conda
pip install CSBDeep
```

For the third part, we will set up a conda enviroment for Noise2Void.

```
# deactivate your active conda environment
conda deactivate
# create a new conda environment called 'n2v' and initialize it with python version 3.7
conda create -n 'n2v' python=3.7
conda activate n2v
conda install tensorflow-gpu=2.4.1 keras=2.3.1 tensorboard nb_conda
pip install n2v
```
