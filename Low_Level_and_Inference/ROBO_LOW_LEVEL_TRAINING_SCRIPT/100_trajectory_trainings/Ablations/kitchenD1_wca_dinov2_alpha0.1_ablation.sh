#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name kitchen-d1-wca-alpha0.1-ablation
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ALPHA ABLATION on KITCHEN_D1 (100 demos, DINOv2, serial WCA) — ALPHA=0.1 variant.
#
# Compared to the top-6 baseline (kitchenD1_weighted_cross_attention_dinov2.sh),
# this run:
#   1. passes many more GMM candidates to the policy (gmm_top_k=TOP_K, default
#      2500 of the 4500 anchors — top-2500 mass is ~100% on every frame, vs
#      top-6 which drops to 13% mass on the most uncertain frames), and
#   2. scales ONLY the WeightedCrossAttention content logits by ALPHA:
#          logits = ALPHA * QK^T/sqrt(d) + log(w_j)
#      Small ALPHA makes the high-level GMM weights dominate candidate ranking
#      (query content can out-rank a candidate only within a weight ratio of
#      ~e^(10*ALPHA) of the top mode). Rough guide:
#          ALPHA=1.0  -> content selects nearly freely (prior = soft bias)
#          ALPHA=0.3  -> effective ~top-6 reachability
#          ALPHA=0.1  -> effective ~top-1..3 (prior nearly decides)
#          ALPHA<=0.01-> prior-only: attention == GMM weights
#      No other attention layer (visual CA / self-attn) is touched.
#
# Launch matrix (env overrides, same pattern as the other scripts):
#   sbatch this_script.sh                       # top-2500, alpha=0.1
#   ALPHA=0.3  sbatch this_script.sh            # top-2500, alpha=0.3
#   USE_ALPHA=false sbatch this_script.sh       # top-2500 CONTROL (original
#                                               #   formula, no alpha) — needed to
#                                               #   separate "more goals" from "alpha"
#   TOP_K=6 USE_ALPHA=false sbatch this_script.sh   # reproduces the old baseline
#
# MEMORY NOTE: goal tokens + per-block WCA K/V activations scale linearly in
# TOP_K * batch_size. At TOP_K=2500 the extra fp32 activation memory is roughly
# ~25 GB at batch 64 and ~50 GB at batch 128 — batch 128 is borderline on an
# 80 GB H100, so BATCH_SIZE defaults to 64 here. If you raise it back to 128
# for strict comparability with the batch-128 baseline, watch the first epoch
# for OOM (and if you *lower* batch instead, consider re-running the top-6
# baseline at the same batch for a fair comparison).

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- ablation knobs -------------------------------------------------------
ALPHA="${ALPHA:-0.1}"          # wca_alpha value (used only when USE_ALPHA=true)
USE_ALPHA="${USE_ALPHA:-true}" # false => original WCA formula (alpha off)
TOP_K="${TOP_K:-2500}"         # gmm_top_k candidates passed to the policy
BATCH_SIZE="${BATCH_SIZE:-64}" # see MEMORY NOTE above
NUM_DEMOS="${NUM_DEMOS:-100}"

if [ "${USE_ALPHA}" = "true" ]; then
    ALPHA_TAG="alpha${ALPHA}"
else
    ALPHA_TAG="noalpha"
fi
RUN_NAME="groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Kitchen_D1_top${TOP_K}_${ALPHA_TAG}_bs${BATCH_SIZE}"

echo "[ablation] TOP_K=${TOP_K}  USE_ALPHA=${USE_ALPHA}  ALPHA=${ALPHA}  BATCH_SIZE=${BATCH_SIZE}"
echo "[ablation] run name: ${RUN_NAME}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_Low_Level_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] files  : demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5  (${NUM_DEMOS} files)"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Generate exactly the demo filenames we want and feed to xargs.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/kitchen_D1_gmm_goal \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.use_alpha="${USE_ALPHA}" \
    policy.wca_alpha="${ALPHA}" \
    policy.gmm_top_k="${TOP_K}" \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name="${RUN_NAME}" \
    name="${RUN_NAME}" \
    training.checkpoint_every=10 \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers=16
