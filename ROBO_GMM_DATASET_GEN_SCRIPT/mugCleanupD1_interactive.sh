#!/bin/bash
# Interactive (no-sbatch) GMM dataset generation for Mug_Cleanup_D1 — ALL demos.
#
# Same pipeline as mugCleanupD1.sh (stage npz -> run high-level GMM -> ship h5
# back to /ocean), with --save_modes_separately so each h5 also gets the
# halo-collapsed obs/gmm_modes + obs/gmm_mode_weights. NO visualization.
#
# Difference from mugCleanupD1.sh: this is meant to be run *inside* an
# already-running interactive allocation (no #SBATCH header, no sbatch needed).
#
# --- HOW TO RUN ----------------------------------------------------------
#   1. Grab an interactive H100 session on the ROBO partition, e.g.:
#        interact -p ROBO --gres=gpu:h100:1 -n 1 --cpus-per-task=12 -t 08:00:00
#   2. Run this script inside that session:
#        bash mugCleanupD1_interactive.sh
#
# It uses $SLURM_JOB_ID (set inside the allocation) for the node-local scratch
# dir, exactly like the batch script.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/Mug_Cleanup_D1_Ellina_Machine"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
CKPT_PATH="${REPO_DIR}/logs/train_Mug_Cleanup_D1_GOAL_SWAP_FULL_1000/2026-05-18/13-48-11/checkpoints/periodic-epoch=epoch=44.ckpt"
FINAL_OCEAN_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/Mug_Cleanup_D1"

# --- node-local scratch --------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/Mug_Cleanup_D1_npz"      # staged inputs
DEST_H5_DIR="${SCRATCH_ROOT}/Mug_Cleanup_D1_gmm_h5"    # converter outputs

# --- (1) stage npz source to /local --------------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_NPZ_DIR}"
echo "[stage] dest   : ${DEST_NPZ_DIR}"
echo "[stage] threads: ${THREADS}"
mkdir -p "${DEST_NPZ_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 ("vanished files") is a benign warning.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DIR_ENV="${SRC_NPZ_DIR}"
export DEST_DIR_ENV="${DEST_NPZ_DIR}"

find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."

# --- (2) run GMM converter, writing h5 (with modes) to /local ------------
mkdir -p "${DEST_H5_DIR}"
cd "${REPO_DIR}"

gen_start=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \
    --dataset_dir "${DEST_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --gmm_output_dir "${DEST_H5_DIR}" \
    --start_demo 0 \
    --max_files 1000 \
    --batch_size 164 \
    --save_modes_separately \
    --mode_radius 0.03 \
    --max_modes 3

gen_elapsed=$(( $(date +%s) - gen_start ))
echo "[gen] done in ${gen_elapsed}s. $(find "${DEST_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l) h5 files written ($(du -sh "${DEST_H5_DIR}" | cut -f1))."

# --- (3) ship the generated h5 files back to /ocean (replace in place) ----
echo "[ship] dest : ${FINAL_OCEAN_DIR}"
mkdir -p "${FINAL_OCEAN_DIR}"

ship_start=$(date +%s)

export SRC_DIR_ENV="${DEST_H5_DIR}"
export DEST_DIR_ENV="${FINAL_OCEAN_DIR}"

find "${DEST_H5_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

ship_elapsed=$(( $(date +%s) - ship_start ))
echo "[ship] done in ${ship_elapsed}s. $(find "${FINAL_OCEAN_DIR}" -maxdepth 1 -name "*.h5" | wc -l) h5 files now in ${FINAL_OCEAN_DIR}."
