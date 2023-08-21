#!/bin/bash -i

# activate base environment
mamba activate base

# create a new environment called '03_image_restoration_part1' and initialize it with python version 3.7
mamba create -y -n 03_image_restoration_part1 python=3.7
# activate the environment
mamba activate 03_image_restoration_part1
# install dependencies from conda
mamba install -y tensorflow-gpu keras jupyter tensorboard nb_conda scikit-image
# install dependencies from pip
pip install CSBDeep
# return to base environment
mamba activate base

# create a new environment called '03_image_restoration_part2'
mamba create -y -n 03_image_restoration_part2 python=3.7
# activate the environment
mamba activate 03_image_restoration_part2
# install dependencies from conda
mamba install -y keras=2.3.1 tensorboard scikit-image nb_conda
# install dependencies from pip
pip install tensorflow-gpu==2.4.1
pip install n2v
# return to base environment
mamba activate base

# create a new environment called '03_image_restoration_bonus'
mamba create -y -n 03_image_restoration_bonus python=3.9
# activate the environment
mamba activate 03_image_restoration_bonus
# install pytorch depencencies
mamba install -y pytorch torchvision pytorch-cuda=11.8 'numpy<1.23' scipy matplotlib tifffile jupyter nb_conda_kernels -c pytorch -c nvidia
# install PPN2V repo from github
pip install git+https://github.com/juglab/PPN2V.git
# activate base environment
mamba activate base


# Data download 
wget https://dl-at-mbl-2023-data.s3.us-east-2.amazonaws.com/image_restoration_data.zip 
unzip -q image_restoration_data.zip