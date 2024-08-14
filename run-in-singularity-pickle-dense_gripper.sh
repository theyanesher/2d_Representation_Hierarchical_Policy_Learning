#!/bin/bash
cd /mnt/RoboGen_sim2real
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:$PYTHONPATH
export PYTHONPATH=${PWD}/rl_games:$PYTHONPATH
export PYTHONPATH=${PWD}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH
export PROJECT_DIR=${PWD}
source prepare.sh
export YUFEI_OPENAI_API_KEY="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" # TODO: embed this in singularity

echo "login to wandb"
WANDB_API_KEY=6bfc6d8e7675422ae39ea6b26d79cfdbc1e92ed3
wandb login $WANDB_API_KEY
echo "login to wandb successed"

echo "start training"
# bash run-act3d-ddp-goal.sh
bash run-act3d-ddp-psc-pickle-dense_gripper.sh train


