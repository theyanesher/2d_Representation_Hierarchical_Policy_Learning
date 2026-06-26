#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # 12 CPU cores for npz load + h5 write workers
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100 (needed by the high-level GMM forward pass)
#SBATCH -t 12:00:00
#SBATCH --job-name kitchen-d1-gmm-gen
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Generate the GMM-annotated h5 dataset for KITCHEN_D1 by running the
# high-level (multimodal) GMM model on every demo's per-step npz, then ship
# the result back to /ocean for the low-level trainer to consume.
#
# Pipeline:
#   1. Stage the source npz tree (D2/KITCHEN_D1) onto the node's /local SSD —
#      many small npz reads from /ocean are slow.
#   2. Run scripts/run_gmm_on_dataset_batch_optimized.py with --gmm_output_dir
#      pointing at a /local h5 dir — h5 writes stay on the fast SSD.
#   3. rsync the generated demo_*.h5 from /local back to the durable /ocean
#      location LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1/.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
CKPT_PATH="${REPO_DIR}/logs/train_KITCHEN_D1_GOAL_SWAP_100demo/2026-06-04/05-12-39/checkpoints/periodic-epoch=epoch=84.ckpt"
FINAL_OCEAN_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1"

# --- node-local scratch --------------------------------------------------
# Per-job isolated subdir so concurrent jobs on the same node never collide.
# PSC's SLURM doesn't always pre-create that subtree, so we force-create it
# ourselves when SLURM_JOB_ID is set.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/KITCHEN_D1_npz"      # staged inputs
DEST_H5_DIR="${SCRATCH_ROOT}/KITCHEN_D1_gmm_h5"    # converter outputs

# --- (1) stage npz source to /local --------------------------------------
# Parallel copy: split the ~1000 top-level demo dirs across N rsync workers
# via xargs -P. Override with RSYNC_THREADS=N env var.
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_NPZ_DIR}"
echo "[stage] dest   : ${DEST_NPZ_DIR}"
echo "[stage] threads: ${THREADS}"
mkdir -p "${DEST_NPZ_DIR}"

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
export SRC_DIR_ENV="${SRC_NPZ_DIR}"
export DEST_DIR_ENV="${DEST_NPZ_DIR}"

# Each rsync handles one top-level entry (a demo_* dir). rsync stays resumable
# per-entry, so re-running the script skips already-copied demos cheaply.
find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."

# --- (2) run GMM converter, writing h5 to /local -------------------------
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
    --batch_size 164

gen_elapsed=$(( $(date +%s) - gen_start ))
echo "[gen] done in ${gen_elapsed}s. $(find "${DEST_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l) h5 files written ($(du -sh "${DEST_H5_DIR}" | cut -f1))."

# --- (3) ship the generated h5 files back to /ocean ---------------------
echo "[ship] dest : ${FINAL_OCEAN_DIR}"
mkdir -p "${FINAL_OCEAN_DIR}"

ship_start=$(date +%s)

# Same per-entry rsync pattern (parallel + resumable). Each entry is one .h5
# file here, but the wrapper handles files and dirs identically.
export SRC_DIR_ENV="${DEST_H5_DIR}"
export DEST_DIR_ENV="${FINAL_OCEAN_DIR}"

find "${DEST_H5_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

ship_elapsed=$(( $(date +%s) - ship_start ))
echo "[ship] done in ${ship_elapsed}s. $(find "${FINAL_OCEAN_DIR}" -maxdepth 1 -name "*.h5" | wc -l) h5 files now in ${FINAL_OCEAN_DIR}."
