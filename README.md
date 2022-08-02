# Exercise 3: Image Restoration

## Setup

Conda environments are useful to keep your workflows running, even if they have conflicting dependencies. For examples,
one method might be using TensorFlow 1.x and the old python 2.7, while another uses TensorFlow 2.x and some python 3
version.

This exercise is split into several parts. They use different libraries so to keep things tidy we will set up separate conda environments. We will make use of the very useful`nb_conda` to be able to switch environments in jupyter.

For the first and second part, we will set up a conda environment for CARE/CSBDeep. Make sure to first deactivate a previous conda environment, e.g. from the last exercise.

```
# create a new conda environment called 'care' and initialize it with python version 3.7
conda create -n care python=3.7
# activate the environment
conda activate care
# install dependencies from conda
conda install tensorflow-gpu keras jupyter tensorboard nb_conda
# install dependencies from pip
pip install CSBDeep
```

For the third part, we will set up a conda enviroment for Noise2Void.

```
# deactivate your active conda environment
conda deactivate
# create a new conda environment called 'n2v' and initialize it with python version 3.7
conda create -n 'n2v' python=3.7
# activate the new environment
conda activate n2v
#install dependencies from conda
conda install tensorflow-gpu=2.4.1 keras=2.3.1 tensorboard nb_conda
# install dependencies from pip
pip install n2v
```

And finally, if you make it to the bonus exercise, we need one more environment. This one using pytorch. In addition, you'll have to clone the ppn2v repo.

```
# deactivate your active conda environment
conda deactivate
# create a new conda environment called 'pn2v'
conda create -n 'ppn2v' python=3.7
# activate the new environment
conda activate ppn2v
# install dependencies from conda
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install nb_conda tifffile matplotlib scipy
# clone the pp2nv repo
git clone https://github.com/juglab/PPN2V.git
```
