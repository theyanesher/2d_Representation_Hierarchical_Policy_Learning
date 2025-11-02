#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 20
#SBATCH --partition=flame
#SBATCH --qos=flame-t3_g3_qos
#SBATCH --gpus=8 #GPU specification. H100
#SBATCH -t 200:00:00 # Estimated time
#SBATCH --job-name articubot-cgn-multivariance
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu


source "/project/flame/yufeiw2/miniconda3/etc/profile.d/conda.sh"
conda activate articubot
cd /project/flame/yufeiw2/RoboGen-sim2real/
source prepare.sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=2    # often helps dataloaders
export MKL_NUM_THREADS=2

bash scripts_orchard/download_articubot_data.sh
# bash scripts_orchard/download_cgn_data.sh

### print all folders in /tmp sorted by newest first
target=/tmp/
echo "Folders in $target (newest first):"
find "$target" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort

cd test_PointNet2

sleep 30000

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=debug \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=debug \
#     general.tasks="['articubot']"
    # cgn.batch_size=15  \
    # cgn.OPTIMIZER.batch_size=15 \
    # general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth


# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_plus_random_cam_plus_close \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=articulated_full_dagger \
#     cgn.batch_size=15  \
#     cgn.OPTIMIZER.batch_size=15 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_load \
#     articubot.batch_size=60   \
#     articubot.num_train_objects=articulated_full_dagger \
#     cgn.batch_size=12  \
#     cgn.OPTIMIZER.batch_size=12 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth