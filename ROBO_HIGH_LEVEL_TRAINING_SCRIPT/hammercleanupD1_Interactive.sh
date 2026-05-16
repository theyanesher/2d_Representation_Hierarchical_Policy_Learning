#!/bin/bash
# Train articubot on HammerCleanupD1.
# Stages the dataset onto the compute node's local scratch ($LOCAL on PSC
# Bridges-2) before training, since reading 900 demos from /ocean is slow.

set -euo pipefail

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/HAMMER_CLEANUP_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide. Fall back to $LOCAL or /tmp only if no
# per-job dir exists (e.g. running outside SLURM).
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/slurm-${SLURM_JOB_ID}/local" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/slurm-${SLURM_JOB_ID}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/HAMMER_CLEANUP_D1"

# --- stage dataset -------------------------------------------------------
# Parallel copy: split the ~900 top-level demo dirs across N rsync workers via
# xargs -P. Override with RSYNC_THREADS=N env var.
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

# Each rsync handles one top-level entry (a demo_* dir). rsync stays resumable
# per-entry, so re-running the script skips already-copied demos cheaply.
find "${SRC_DATA_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/train.py \
    model=articubot \
    dataset=hammerCleanupD1 \
    dataset.data_dir="${DEST_DATA_DIR}" \
    model.use_rgb=False \
    model.in_channels=4 \
    training.batch_size=128 \
    wandb.entity=pbhowal-carnegie-mellon-university \
    "hydra.run.dir=logs/train_HammerCleanup_D1_GOAL_SWAP/$(date +%Y-%m-%d/%H-%M-%S)" \
    +dataset.use_weighted_sampler=True \
    +dataset.transition_p=0.5 \
    +dataset.transition_radius=5 \
    +dataset.transition_label_swap=True \
    +dataset.transition_swap_p_max=0.5 \
    "resources.gpus=[0]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15
