#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # 12 CPU cores for npz load + npz write workers
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100 (needed by the high-level GMM forward pass)
#SBATCH -t 2:00:00
#SBATCH --job-name coffee-prep-d1-awe-pred-gen
#SBATCH -o /ocean/projects/cis240052p/eswaramo/code/2d_Representation_Hierarchical_Policy_Learning/logs/coffee-prep-d1-gmm-awe-subgoals-100demo_%j.out
#SBATCH -e /ocean/projects/cis240052p/eswaramo/code/2d_Representation_Hierarchical_Policy_Learning/logs/coffee-prep-d1-gmm-awe-subgoals-100demo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# Generate per-frame GMM prediction npz files for COFFEE_PREPERATION_D1 by running the
# RDP-goal-trained high-level articubot model on every demo's per-step npz.
# UNLIKE the classic GMM h5 generation, this writes ONLY the model's outputs
# (gmm_pred_goal_rdp / gmm_all_goals_rdp / gmm_all_weights_rdp) as a parallel
# npz tree — no h5 files are touched.
#
# Pipeline:
#   1. Stage the source npz tree onto the node's /local SSD.
#   2. Run scripts/run_gmm_pred_to_npz.py writing to a /local npz dir.
#   3. rsync the prediction tree back to the durable /ocean location
#      EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1_GMM_PRED/ (with a free-space guard first —
#      first-100 tree is ~6-7 GB; a full-1000 tree would be 50-70 GB).
#
# CKPT_PATH below is set EXPLICITLY (same convention as the other gen
# scripts) — edit the path to switch checkpoints. The script fails fast if
# the file does not exist yet (e.g. the RDP training has not finished).

set -euo pipefail
set -x

export PIXI_HOME="/ocean/projects/cis240052p/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

GOAL_SOURCE="awe"

# --- demo selection ------------------------------------------------------
# Only the first NUM_DEMOS demos are staged + processed: the low-level WCA
# experiments train on the first 100 demos only. Lift later with e.g.:
#   NUM_DEMOS=1000 sbatch this_script.sh   (per-demo resume skips existing)
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/jet/home/eswaramo/data/D2/COFFEE_PREPERATION_D1"
REPO_DIR="/jet/home/eswaramo/code/2d_Representation_Hierarchical_Policy_Learning"
FINAL_OCEAN_DIR="/jet/home/eswaramo/data/D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1_AWE_GMM_PRED"

# --- checkpoint (EDIT HERE to change) --------------------------------------
CKPT_PATH="${REPO_DIR}/logs/train_COFFEE_PREPERATION_D1_AWE_subgoals_100demo/2026-08-10/16-58-37/checkpoints/epoch=59-step=22500-val/rmse=0.026.ckpt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    echo "[ckpt] Edit CKPT_PATH in this script (training not finished yet?)." >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

# --- node-local scratch ----------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/COFFEE_PREPERATION_D1_npz"        # staged inputs
DEST_PRED_DIR="${SCRATCH_ROOT}/COFFEE_PREPERATION_D1_GMM_PRED"  # generator outputs

# --- (1) stage npz source to /local ----------------------------------------
THREADS="${RSYNC_THREADS:-32}"
echo "[stage] source : ${SRC_NPZ_DIR}"
echo "[stage] dest   : ${DEST_NPZ_DIR}"
mkdir -p "${DEST_NPZ_DIR}"
stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DIR_ENV="${SRC_NPZ_DIR}"
export DEST_DIR_ENV="${DEST_NPZ_DIR}"

seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."

# --- (2) run the prediction generator, writing npz to /local ----------------
mkdir -p "${DEST_PRED_DIR}"
cd "${REPO_DIR}"
gen_start=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
# PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_pred_to_npz.py \
    --dataset_dir "${DEST_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${DEST_PRED_DIR}" \
    --key_suffix "${GOAL_SOURCE}" \
    --start_demo 0 \
    --max_files "${NUM_DEMOS}" \
    --batch_size 164

gen_elapsed=$(( $(date +%s) - gen_start ))
echo "[gen] done in ${gen_elapsed}s. $(find "${DEST_PRED_DIR}" -name '*.npz' | wc -l) npz files written ($(du -sh "${DEST_PRED_DIR}" | cut -f1))."

# --- (3) ship back to /ocean, with a free-space guard -----------------------
need_kb=$(du -sk "${DEST_PRED_DIR}" | cut -f1)
avail_kb=$(df -k --output=avail "$(dirname "${FINAL_OCEAN_DIR}")" | tail -1 | tr -d ' ')
buffer_kb=$(( 20 * 1024 * 1024 ))  # keep >=20GB headroom on /ocean after shipping
if [ "$(( need_kb + buffer_kb ))" -gt "${avail_kb}" ]; then
    echo "[ship] ERROR: need ${need_kb}KB + 20GB buffer but only ${avail_kb}KB free on /ocean." >&2
    echo "[ship] Prediction tree is preserved on ${DEST_PRED_DIR} for THIS job only — free space and re-run (generation resumes instantly via per-demo skip)." >&2
    exit 1
fi

echo "[ship] dest : ${FINAL_OCEAN_DIR}"
mkdir -p "${FINAL_OCEAN_DIR}"
ship_start=$(date +%s)
export SRC_DIR_ENV="${DEST_PRED_DIR}"
export DEST_DIR_ENV="${FINAL_OCEAN_DIR}"
find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}
ship_elapsed=$(( $(date +%s) - ship_start ))
echo "[ship] done in ${ship_elapsed}s. $(find "${FINAL_OCEAN_DIR}" -name '*.npz' | wc -l) npz files now in ${FINAL_OCEAN_DIR}."
