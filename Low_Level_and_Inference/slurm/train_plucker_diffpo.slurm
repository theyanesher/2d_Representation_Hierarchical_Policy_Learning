#!/bin/bash
#SBATCH --job-name=plucker_diffpo
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mnakuraf@andrew.cmu.edu

# --- 1. Setup Environment & Paths ---
PROJECT_DIR="/project/flame/mnakuraf/ArticuBot"
SCRATCH_DIR="/tmp/${SLURM_JOB_ID}"
TIMESTAMP=$(date +%Y.%m.%d_%H.%M.%S)

# Training Outputs
TRAIN_OUTPUT_DIR="${SCRATCH_DIR}/outputs/${TIMESTAMP}_train"
# Input Test Configs (extracted from data.tar)
TEST_CONFIG_DIR="${SCRATCH_DIR}/data/rgb_eval"

cd $PROJECT_DIR

# --- 2. Prepare Data ---
mkdir -p "${SCRATCH_DIR}/data"
echo "Fetching and extracting data..."
gsutil cp gs://cmu-gpucloud-mnakuraf/rgb-articubot/data.tar "${SCRATCH_DIR}/data/"
tar -xf "${SCRATCH_DIR}/data/data.tar" -C "${SCRATCH_DIR}/data"

# --- 3. Training ---
echo "Starting Training..."
pixi run python diffusion_policy/train.py \
    --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=plucker_articubot \
    task.dataset.data_dir="${SCRATCH_DIR}/data/rgb_camera_left_right_randomized/41510" \
    hydra.run.dir="${TRAIN_OUTPUT_DIR}" \
    encoder_backbone=shared_crop

# --- 6. Package & Upload ---
echo "Uploading results to GCS..."
ARCHIVE="${SCRATCH_DIR}/${TIMESTAMP}_train.tar.gz"
# We archive the training dir which now contains the nested eval results
tar -czf "${ARCHIVE}" -C "${TRAIN_OUTPUT_DIR}" .

gsutil cp "${ARCHIVE}" "gs://cmu-gpucloud-mnakuraf/rgb-articubot/outputs/"

# --- 7. Cleanup ---
rm -rf "${SCRATCH_DIR}"
echo "Job finished at $(date)"
