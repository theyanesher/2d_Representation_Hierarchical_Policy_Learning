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
export YUFEI_OPENAI_API_KEY="xxx" # TODO: embed this in singularity

echo "start training"
cd test_PointNet2

# torchrun --standalone --nproc_per_node=1 train_ddp_weighted_displacement.py --batch_size 170 \
#     --num_epochs 300 --model_type pointnet2_large --model_invariant \
#     --exp_path /ocean/projects/cis240052p/ywang59/RoboGen_sim2real/test_PointNet2/exps \
#     --num_train_objects 300 --dataset_prefix /local --predict_two_goals --exp_name train_300_epochs

torchrun --standalone --nproc_per_node=1 train_ddp_weighted_displacement.py --batch_size 170 \
    --num_epochs 500 --model_type pointnet2_large --model_invariant \
    --exp_path /ocean/projects/cis240052p/ywang59/RoboGen_sim2real/test_PointNet2/exps \
    --num_train_objects 500 --dataset_prefix /local --predict_two_goals --exp_name train_500_epochs


    