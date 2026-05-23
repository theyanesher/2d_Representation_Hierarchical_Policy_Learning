#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name mug-cleanup-d1-goal-gripper-parallel-ca
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Train flow-matching DiT low-level policy on Mug_Cleanup_D1 conditioned on
# the single ground-truth goal_gripper_pts (4 keypoints), with PARALLEL goal-
# and visual-cross-attention inside each DiT cross-attn block (both read the
# pre-block hidden_states, outputs are concat'd along the feature dim and
# projected back to D via fuse_proj before the residual add).
#
# The trainer's LazyArticuBotDataset reads demo_*.h5 shards, but the source on
# /ocean is the GROOT-style demo_N/ npz tree, so we first consolidate it into
# h5 (idempotent — skips already-converted demos), then stage the h5 files
# onto node-local /local scratch before training. No checkpoint resume.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/Mug_Cleanup_D1_Ellina_Machine"
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/Mugcleanup_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- generate npz -> h5 if missing ---------------------------------------
# generate_non_gmm_goals_for_low_level.py with --no_gmm consolidates each
# demo_N/<t>.npz directory into one demo_N.h5 file in --no_gmm_output_dir.
# It is per-demo idempotent (line 317-320 in the script). We additionally
# skip the python startup entirely when every demo already has its h5.
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"

if [ "${src_demo_count}" -gt 0 ] && [ "${existing_h5_count}" -ge "${src_demo_count}" ]; then
    echo "[gen] all ${src_demo_count} demos already converted → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (fills only missing files)"
    mkdir -p "${NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${NO_GMM_H5_DIR}"
    )
    echo "[gen] done. *.h5 count now: $(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l)"
fi

# --- node-local scratch --------------------------------------------------
# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide. PSC's SLURM doesn't always pre-create
# that subtree, so we force-create it ourselves when SLURM_JOB_ID is set. Fall
# back to $LOCAL or /tmp only if no per-job dir exists.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
SRC_DATA_DIR="${NO_GMM_H5_DIR}"
DEST_DATA_DIR="${SCRATCH_ROOT}/Mug_Cleanup_D1_Low_Level"

# --- stage dataset -------------------------------------------------------
# Parallel copy: split the top-level demo_*.h5 shards across N rsync workers
# via xargs -P. Override with RSYNC_THREADS=N env var.
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 ("vanished files") is a benign warning —
# usually rsync's own temp files (.<name>.XXXXXX) left behind by a previously
# interrupted sync. Treat it as success so it doesn't trip xargs/set -e.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Each rsync handles one top-level entry (a demo_*.h5 file). rsync stays
# resumable per-entry, so re-running the script skips already-copied files
# cheaply.
find "${SRC_DATA_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Parallel cross-attention prerequisites:
#   policy.use_goal_cross_attention=true     — route goal_gripper_pts through the
#                                              DiT's dedicated goal cross-attn module
#                                              (rather than prepending to hidden_states
#                                              for self-attention).
#   policy.use_parallel_cross_attentions=true — goal-CA and visual-CA both read the
#                                              pre-block hidden_states; their outputs
#                                              are concat'd and fused via Linear(2D, D).
# use_weighted_cross_attention stays false (no GMM in this dataset).
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/mug_cleanup_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_parallel_cross_attentions=true \
    logging.project=mimicgen_tasks \
    logging.name=mug_cleanup_D1_goal_gripper_parallel_CA_DIT \
    name=mug_cleanup_D1_goal_gripper_parallel_CA_DIT \
    dataloader.batch_size=164 \
    dataloader.num_workers=32 \
    training.checkpoint_every=10
