#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 16:00:00 # Estimated time, 16hour max. DD-HH:MM.
#SBATCH --job-name kitchen-d1-high-level-100demo-sigmoid-swap-tau0.5
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ABLATION: SIGMOID transition-swap profile on KITCHEN_D1.
#
# Identical to ../../kitchenD1.sh EXCEPT the goal label-swap augmentation uses
# the SIGMOID profile instead of the default linear/triangular one:
#
#   linear (baseline run):  p_swap(d) = p_max * (1 - d/(radius+1))
#                           hard window at +-transition_radius
#   sigmoid (this run):     p_swap(d) = 2 * p_max * sigmoid(-d / tau)
#                           = p_max at the transition, smooth decay, NO window
#
# Reframed as P(label = next goal), the sigmoid gives a smooth S-ramp from ~0
# before the transition to ~1 after it (50/50 at the boundary), replacing the
# piecewise-linear ramp of the baseline. transition_radius is still passed but
# is used ONLY by the weighted sampler (which frames get 50% sampling mass) —
# the sigmoid swap itself reads only tau.
#
# TAU defaults to 1.25 (= radius/4: the sigmoid effectively saturates at the
# same +-5-frame scale as the baseline triangle, sigma(-4) ~ 0.018). Override:
#   TAU=2.0 sbatch this_script.sh
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- sigmoid temperature (overridable) ------------------------------------
TAU="${TAU:-0.5}"
echo "[sigmoid_swap] tau=${TAU}  (temperature of the swap-probability sigmoid)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/KITCHEN_D1"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS demo dirs) ----------------------------
# Parallel copy: split the requested demo dirs across N rsync workers via
# xargs -P. Override with RSYNC_THREADS=N env var.
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] dirs   : demo_0 .. demo_$((NUM_DEMOS-1))  (${NUM_DEMOS} dirs)"
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

# Generate exactly the demo dir names we want and feed to xargs. Each rsync
# handles one top-level demo_* dir and stays resumable per-entry, so re-running
# the script skips already-copied demos cheaply.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} dirs, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} demo dirs staged, got ${staged_count}." >&2
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
    dataset=Kitchen \
    dataset.data_dir="${DEST_DATA_DIR}" \
    model.use_rgb=False \
    model.in_channels=4 \
    training.batch_size=164 \
    wandb.entity=pbhowal-carnegie-mellon-university \
    "hydra.run.dir=logs/ABLATION/SIGMOID_EXPERIMENTS/KITCHEN_D1/train_KITCHEN_D1_GOAL_SWAP_SIGMOID_tau${TAU}_${NUM_DEMOS}demo/$(date +%Y-%m-%d/%H-%M-%S)" \
    +dataset.use_weighted_sampler=True \
    +dataset.transition_p=0.5 \
    +dataset.transition_radius=5 \
    +dataset.transition_label_swap=True \
    +dataset.transition_swap_p_max=0.5 \
    +dataset.transition_swap_profile=sigmoid \
    +dataset.transition_swap_tau="${TAU}" \
    "resources.gpus=[0]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15
