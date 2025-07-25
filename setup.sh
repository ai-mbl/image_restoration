#!/bin/bash

# >>> Conda initialization <<<
source ~/conda/etc/profile.d/conda.sh # FIXME: this only works for MBL course machine

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
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install git+https://github.com/CAREamics/MicroSplit-reproducibility.git
    pip install tensorboard
    # Using pytorch-lightning 2.4.0 causes bugs in tensorboard and interupting training.
    pip install pytorch-lightning==2.3.3
    pip install git+https://github.com/dlmbl/dlmbl-unet
    python -m ipykernel install --user --name "05_image_restoration"
    # Clone the extra repositories
    git clone https://github.com/krulllab/COSDD.git -b n_dimensional 03_COSDD/COSDD
fi

# Download the data
CARE + N2V
python download_careamics_portfolio.py
cd data/
# COSDD
wget "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100888/03-mito-confocal/mito-confocal-lowsnr.tif"
mkdir CCPs/
cd CCPs/
gdown 16oiMkH3cpVU500MSPbm7ghOpEMoD2YNu
cd ../
mkdir ER/
cd ER/
gdown 1Bho6Oymfxi7OV0tPb9wkINkVOCpTaL7M
cd ../
mkdir Microtubules/
cd Microtubules/
gdown 14sPIEE2qU2J6oRFMz46v2IvkCVjFX8D1
cd ../
mkdir F-actin/
cd F-actin/
gdown 1FYO-Bpl5vjpiJ6kzV1qO1pL37Y3Dirfy
cd ../../
mkdir 03_COSDD/checkpoints
cd 03_COSDD/checkpoints
gdown --folder 1_oUAxagFVin71xFASb9oLF6pz20HjqTr
cd ../../
# MicroSplit
wget https://download.fht.org/jug/MicroSplit_MBL_2025.zip
unzip MicroSplit_MBL_2025.zip -d 04_MicroSplit/
rm MicroSplit_MBL_2025.zip