#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name sweep-to-dustpan-high-level
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/RL_BENCH_TRAINING_SCRIPTS/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/RL_BENCH_TRAINING_SCRIPTS/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Train articubot (GMM cross-displacement high-level policy) on the RL Bench
# sweep_to_dustpan_of_size task. RL Bench is handled via the rl_bench dataset
# group (is_rl_bench=True): `front` is the primary "agentview" camera and the
# scene point_cloud is the front+left+right fusion (wrist ignored).
#
# Stages the dataset onto the compute node's local scratch ($LOCAL on PSC
# Bridges-2) before training, since reading many demos from /ocean is slow.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
# Source points at the directory that directly contains the demo_* dirs.
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/RL_BENCH_DATASETS/sweep_to_dustpan_of_size/sweep_to_dustpan_of_size"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"

# Pick a node-local scratch dir. We want the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so concurrent jobs on the same node never
# collide. PSC's SLURM doesn't always pre-create that subtree on this cluster,
# so we force-create it ourselves when SLURM_JOB_ID is set.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/sweep_to_dustpan_of_size"

# --- one-time: add gripper_pcd to the source dataset (Option 2) ----------
# RL Bench .npz frames lack `gripper_pcd` (the 4 current-gripper keypoints the
# training loop reads as action_pcd). We add it once, permanently, into the
# canonical /ocean source (atomic per-file) BEFORE staging, so the staged copy
# and all future runs already have it and the dataloader has zero extra cost.
# Guarded by a marker file so this runs only on the very first launch.
GRIPPER_MARKER="${SRC_DATA_DIR}/.gripper_pcd_added"
if [ ! -f "${GRIPPER_MARKER}" ]; then
    echo "[gripper_pcd] marker absent -> generating once into source (permanent)..."
    ( cd "${REPO_DIR}" && \
      PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
      PYTHONNOUSERSITE=1 \
      USE_TF=0 \
      pixi run python \
        "${REPO_DIR}/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/RL_BENCH_TRAINING_SCRIPTS/add_gripper_pcd.py" \
        --data_dir "${SRC_DATA_DIR}" )
else
    echo "[gripper_pcd] already present (marker found) -> skipping generation."
fi

# --- stage dataset -------------------------------------------------------
# Parallel copy: split the top-level demo dirs across N rsync workers via
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

# NOTE: the dataset config lives in a subfolder (rl_bench_datasets/), so the
# `training: ${dataset}_${model}` interpolation in configs/train.yaml would
# resolve to training/rl_bench_datasets/... — which doesn't exist. We pass
# `training=` explicitly to bypass that interpolation.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/train.py \
    model=articubot \
    dataset=rl_bench_datasets/sweep_to_dustpan_of_size \
    training=sweep_to_dustpan_of_size_articubot \
    dataset.data_dir="${DEST_DATA_DIR}" \
    dataset.add_language_cond=True \
    model.use_rgb=False \
    model.in_channels=4 \
    training.batch_size=164 \
    wandb.entity=pbhowal-carnegie-mellon-university \
    "hydra.run.dir=logs/train_Sweep_To_Dustpan_Of_Size_GOAL_SWAP_FULL_LANGCOND/$(date +%Y-%m-%d/%H-%M-%S)" \
    +dataset.use_weighted_sampler=True \
    +dataset.transition_p=0.5 \
    +dataset.transition_radius=5 \
    +dataset.transition_label_swap=True \
    +dataset.transition_swap_p_max=0.5 \
    "resources.gpus=[0]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15
