#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 1
#SBATCH -p preempt
#SBATCH --gpus=4 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name articubot_alone_in_articulated
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu

source "/project/flame/yufeiw2/miniconda3/etc/profile.d/conda.sh"
conda activate articubot
cd /project/flame/yufeiw2/RoboGen-sim2real/
source prepare.sh
cd test_PointNet2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
    general.exp_name=articubot_cgn_in_not_fp_to_full \
    articubot.batch_size=70 \
    cgn.batch_size=15 \
    cgn.OPTIMIZER.batch_size=15 \
    general.fp_to_full=0 \
    general.first_sa_point=2048 \
    cgn.DATA.further_fps_point=0 \


# torchrun --standalone --nproc_per_node=4 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
#     general.exp_name=articubot_alone_in_articulated \
#     general.tasks="['articubot']" \
#     articubot.batch_size=140 \

# torchrun --standalone --nproc_per_node=5 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs \
    # general.exp_name=articubot_alone_in_articulated_gmm \
    # general.tasks="['articubot']" \
    # articubot.batch_size=130 \
    # articubot.gmm=1 \