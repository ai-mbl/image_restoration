#!/bin/bash

# create environment
ENV="05_image_restoration"
conda create -y -n "$ENV" python=3.10
conda activate "$ENV"

# check that the environment was activated
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    echo "Environment activated successfully"
else
    echo "Failed to activate the environment"
fi

# Further instructions that should only run if the environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENV" ]]; then
    conda install -y pytorch-gpu cuda-toolkit=11.8 torchvision -c nvidia -c conda-forge -c pytorch
    #mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install ipykernel "careamics[examples,tensorboard] @ git+https://github.com/CAREamics/careamics.git"
    python -m ipykernel install --user --name "05_image_restoration"
fi
