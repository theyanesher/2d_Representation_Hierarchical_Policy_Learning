#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name sweep-dustpan-wca-100demo-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/RL_TRAINING_DATASET/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/RL_TRAINING_DATASET/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo RL Bench GMM weighted-cross-attention run (NATIVE 8-D state/action).
#
# GMM counterpart of sweep_to_dustpan_goal_gripper.sh: trains the same
# flow-matching DiT low-level policy on the RL Bench sweep_to_dustpan_of_size
# task, but conditioned on the FULL high-level GMM goal distribution
# (gmm_all_goals + gmm_all_weights) via weighted goal cross-attention — instead
# of the single ground-truth goal_gripper_pts. Mirrors the MimicGen
# mugcleanupD1_weighted_cross_attention_dinov2.sh recipe, adapted for RL Bench:
#   - 4 cameras (front/left_shoulder/right_shoulder/wrist) at 128x128
#   - native 8-D state/action (pos + quaternion + gripper), action_mode=hybrid_delta
#   - DINOv2 crop 126x126 (9*14; the 224/256 analog for 128px input)
#   - WCA flags: use_goal_cross_attention + use_weighted_cross_attention, gmm_top_k=64
#
# Unlike the goal_gripper baseline, this script does NOT build its dataset: the
# GMM-annotated h5 is produced separately by the high-level converter job
# (ROBO_GMM_DATASET_GEN_SCRIPT/sweepToDustpanOfSize.sh). This script just stages
# the first NUM_DEMOS demos to node-local scratch and trains.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
# Permanent GMM-annotated h5 dataset on /ocean, produced by the high-level
# converter job (ROBO_GMM_DATASET_GEN_SCRIPT/sweepToDustpanOfSize.sh).
# Overridable via the RL_BENCH_GMM_H5_DIR env var.
SRC_DATA_DIR="${RL_BENCH_GMM_H5_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/RL_BENCH_DATASETS/sweep_to_dustpan_of_size}"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- precondition: GMM dataset must already exist ------------------------
# This trainer consumes the GMM h5; it does not generate it. Fail early (with a
# pointer to the generator) rather than mid-stage if the dataset is missing or
# has fewer than NUM_DEMOS demos.
if [ ! -d "${SRC_DATA_DIR}" ]; then
    echo "[precheck] ERROR: GMM dataset dir not found: ${SRC_DATA_DIR}" >&2
    echo "[precheck] Generate it first with ROBO_GMM_DATASET_GEN_SCRIPT/sweepToDustpanOfSize.sh" >&2
    exit 1
fi
avail_h5=$(find "${SRC_DATA_DIR}" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l)
echo "[precheck] GMM h5 available in ${SRC_DATA_DIR}: ${avail_h5}"
if [ "${avail_h5}" -lt "${NUM_DEMOS}" ]; then
    echo "[precheck] ERROR: need ${NUM_DEMOS} demos but only ${avail_h5} h5 present." >&2
    exit 1
fi

# --- node-local scratch --------------------------------------------------
# Always prefer the per-job isolated subdir (/local/slurm-<jobid>/local/) so
# SLURM auto-cleans on job end and concurrent jobs never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Sweep_To_Dustpan_GMM_Low_Level_${NUM_DEMOS}demo"

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
    task=RL_Bench_Tasks/sweep_to_dustpan_gmm_goal \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    "policy.crop_shape=[126,126]" \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=64 \
    logging.project=rl_bench_tasks \
    logging.name=sweep_to_dustpan_gmm_wca_${NUM_DEMOS}demo_dinov2_DIT \
    name=sweep_to_dustpan_gmm_wca_${NUM_DEMOS}demo_dinov2_DIT \
    dataloader.batch_size=64 \
    dataloader.num_workers=16 \
    training.checkpoint_every=10
