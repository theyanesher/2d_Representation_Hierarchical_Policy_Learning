#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # 12 CPU cores for npz load + h5 write workers
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100 (needed by the high-level GMM forward pass)
#SBATCH -t 4:00:00
#SBATCH --job-name sweep-to-dustpan-gmm-gen
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Generate the GMM-annotated h5 dataset for the RLBench sweep_to_dustpan_of_size
# task by running the high-level (multimodal) GMM model on every demo's per-step
# npz, then ship the result back to /ocean for the low-level trainer to consume.
#
# This is the RLBench sibling of mugCleanupD1.sh. The ONLY behavioral difference
# is the --rl_bench flag passed to the converter: instead of the fixed MimicGen
# 2-camera schema (cam0/cam1), the adapter copies EVERY original npz key verbatim
# into obs/<key> (all 4 RLBench cameras: front / left_shoulder / right_shoulder /
# wrist, plus eef_pos, eef_quat, gripper_qpos, lang_goal, episode_idx, reward,
# terminal, timeout, ...). The three obs/gmm_* fields and action/delta +
# action/hybrid are added exactly as in the MimicGen path.
#
# Pipeline:
#   1. Stage the source npz tree onto the node's /local SSD.
#   2. Run scripts/run_gmm_on_dataset_batch_optimized.py --rl_bench with
#      --gmm_output_dir pointing at a /local h5 dir.
#   3. rsync the generated demo_*.h5 from /local back to the durable /ocean dir.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
# NOTE: the source path is doubly nested; point at the INNER dir that holds demo_*.
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/RL_BENCH_DATASETS/sweep_to_dustpan_of_size/sweep_to_dustpan_of_size"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
# Language-conditioned high-level checkpoint (trained with dataset.add_language_cond=True).
# Must be paired with the --use_lang_goal converter flag below so each demo is
# conditioned on its own lang_goal (tall vs short dustpan).
CKPT_PATH="${REPO_DIR}/logs/train_Sweep_To_Dustpan_Of_Size_GOAL_SWAP_FULL_LANGCOND/2026-06-25/20-57-42/checkpoints/periodic-epoch=epoch=89.ckpt"
# Separate _LANGCOND output dir so we don't overwrite the old (zeros-conditioned)
# generated dataset. Point the low-level trainer at this path.
FINAL_OCEAN_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/RL_BENCH_DATASETS/sweep_to_dustpan_of_size_LANGCOND"

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
DEST_NPZ_DIR="${SCRATCH_ROOT}/sweep_to_dustpan_npz"      # staged inputs
DEST_H5_DIR="${SCRATCH_ROOT}/sweep_to_dustpan_gmm_h5"    # converter outputs

# --- (1) stage npz source to /local --------------------------------------
# Parallel copy: split the top-level demo dirs across N rsync workers via
# xargs -P. Override with RSYNC_THREADS=N env var.
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
# -mindepth/-maxdepth 1 also picks up the top-level .gripper_pcd_added marker
# file; that is harmless (the converter only processes demo_* directories).
find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."

# --- (2) run GMM converter (RLBench adapter), writing h5 to /local -------
mkdir -p "${DEST_H5_DIR}"
cd "${REPO_DIR}"

gen_start=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \
    --rl_bench \
    --use_lang_goal \
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
