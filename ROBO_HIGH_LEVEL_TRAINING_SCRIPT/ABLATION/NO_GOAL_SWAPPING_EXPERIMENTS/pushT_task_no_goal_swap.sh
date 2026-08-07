#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 8:00:00 # Estimated time. DD-HH:MM.
#SBATCH --job-name push-t-task-high-level-no-goal-swap
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ABLATION: NO goal-swap augmentation on PushT_Task (standard GMM training).
#
# Identical to ../../pushT_task.sh (all 206 demos, has_camera=False dataset,
# PushT-scaled fixed_variance) EXCEPT the transition machinery is fully
# removed (same convention as the *_NO_SWAP_STANDARD_GMM.sh scripts):
#   - NO transition_label_swap (goals are never flipped near transitions)
#   - NO use_weighted_sampler (uniform frame sampling, no transition
#     oversampling)
# Everything else (model, data, batch size, epochs, checkpoints) matches the
# GOAL_SWAP baseline, so the comparison isolates the full goal-swap recipe.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/PushT_Task"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/PushT_Task"

# --- stage dataset (all demos) -------------------------------------------
# Parallel copy: split the demo dirs across N rsync workers via xargs -P.
# Override with RSYNC_THREADS=N env var. Whole dataset is only ~1 GB.
THREADS="${RSYNC_THREADS:-32}"

src_demo_count=$(find "${SRC_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
echo "[stage] source : ${SRC_DATA_DIR} (${src_demo_count} demos)"
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

# Sanity check: every staged demo dir must actually contain .npz frames.
staged_count=$(find "${DEST_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
nonempty_count=$(
    find "${DEST_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} dirs (${nonempty_count} with npz), $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${src_demo_count}" ] || [ "${nonempty_count}" -ne "${src_demo_count}" ]; then
    echo "[stage] ERROR: source has ${src_demo_count} demos; staged ${staged_count} (${nonempty_count} with npz)." >&2
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
pixi run python scripts/train.py \
    model=articubot \
    dataset=push_t_task \
    dataset.data_dir="${DEST_DATA_DIR}" \
    model.use_rgb=False \
    model.in_channels=4 \
    "model.fixed_variance=[0.0005,0.002,0.005,0.01,0.02]" \
    training.batch_size=128 \
    wandb.entity=pbhowal-carnegie-mellon-university \
    "hydra.run.dir=logs/ABLATION/NO_GOAL_SWAPPING_EXPERIMENTS/PUSH_T_TASK/train_PushT_Task_NO_GOAL_SWAP/$(date +%Y-%m-%d/%H-%M-%S)" \
    "resources.gpus=[0]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15
