#!/bin/bash


cd /mnt/RoboGen-sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
export WANDB_API_KEY=c9187c7dfcc339af75f2f47c3b80c95743057b42

source prepare.sh
python 3d_diffusion_policy/merge_trajectories.py

