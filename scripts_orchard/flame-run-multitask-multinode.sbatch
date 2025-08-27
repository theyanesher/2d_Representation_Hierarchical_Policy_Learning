#!/bin/bash

#SBATCH -J articubot-cgn-multivariance
#SBATCH -N 2                         # two nodes
#SBATCH --ntasks-per-node=1          # one launcher per node
#SBATCH --gpus-per-node=8            # 8 GPUs on each node
#SBATCH --cpus-per-gpu=4          # tune for your dataloaders
#SBATCH --partition=flame
#SBATCH --qos=flame-t3_g3_qos
#SBATCH -t 200:00:00
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu

# --- Activate conda (avoid nounset issues around activation) ---
source "/project/flame/yufeiw2/miniconda3/etc/profile.d/conda.sh"
conda activate articubot

# --- Env / tuning ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Optional NCCL hints (uncomment/tailor to your cluster)
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0   # or eth0/enp* etc.

ROOT=/project/flame/yufeiw2/RoboGen-sim2real
cd "$ROOT"
source prepare.sh

# Use node-local scratch if Slurm provides it, otherwise /tmp
export TARGET="/tmp/"

# --- Stage data on EVERY node's local scratch ---
# If your download scripts can read $TARGET, great.
# If not, they likely default to /tmp, which is also fine since TARGET==/tmp fallback.
srun --ntasks-per-node=1 --export=ALL bash -lc '
  # propagate PATH/LD_*/MKL vars
  cd /project/flame/yufeiw2/RoboGen-sim2real
  export TARGET="/tmp/"

  echo "[$HOSTNAME] Staging data to $TARGET ..."
  bash scripts_orchard/download_articubot_data2.sh
  bash scripts_orchard/download_cgn_data2.sh

  echo "[$HOSTNAME] Folders in $TARGET (mtime newest first):"
  ls -1dt "$TARGET"/*/ 2>/dev/null | xargs -r -n1 basename
'

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

GPUS_PER_NODE=8
export WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))

cd test_PointNet2

srun --ntasks-per-node=1 --ntasks=${SLURM_NNODES} \
  torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
    general.exp_name=test_multinode_training \
    articubot.batch_size=80   \
    articubot.num_train_objects=aritucbot_new_cat_camera_random_close \
    cgn.batch_size=15  \
    cgn.OPTIMIZER.batch_size=15 



# srun --ntasks-per-node=1 --ntasks=${SLURM_NNODES} \
#   torchrun \
#     --nnodes="${SLURM_NNODES}" \
#     --nproc_per_node="${GPUS_PER_NODE}" \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#     train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=test_multinode_training \
#     articubot.batch_size=70   \
#     articubot.num_train_objects=debug \
#     general.tasks="['articubot']" \
#     cgn.batch_size=15  \
#     cgn.OPTIMIZER.batch_size=15 \

# torchrun --standalone --nproc_per_node=8 train_multitask_ddp_weighted_displacement_gmm.py --arg_configs    \
#     general.exp_name=articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger_load \
#     articubot.batch_size=60   \
#     articubot.num_train_objects=articulated_full_dagger \
#     cgn.batch_size=12  \
#     cgn.OPTIMIZER.batch_size=12 \
#     general.load_model_path=/project/flame/yufeiw2/RoboGen-sim2real/test_PointNet2/exps/2025-08-14articubot_cgn_ln_gmm_multivariance_full_450_plus_dagger/model_300000.pth