#!/bin/bash

ROOT=${PWD} 

### create conda environment ###
conda create -y -n here python=3.9 cmake=3.25.0 -c conda-forge

### activate conda environment ###
source ~/anaconda3/etc/profile.d/conda.sh
conda activate here

### Setup habitat-sim ###
cd ${ROOT}/third_parties
git clone https://github.com/facebookresearch/habitat-sim
cd habitat-sim
git checkout v0.3.1
pip install -r requirements.txt
python setup.py install --headless --bullet

### extra installation ###
pip install numpy==1.23.5
pip install opencv-python
conda install -y ipython
pip install mmcv==2.0.0

### CoSLAM installation ###
cd ${ROOT}/third_parties/coslam
git checkout 3bb904e
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd external/NumpyMarchingCubes
python setup.py install

### NARUTO installation ###
pip install -r ${ROOT}/envs/requirements.txt
