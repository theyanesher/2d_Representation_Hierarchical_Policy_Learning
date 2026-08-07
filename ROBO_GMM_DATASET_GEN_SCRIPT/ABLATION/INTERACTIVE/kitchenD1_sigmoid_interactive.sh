#!/bin/bash
# INTERACTIVE variant of ../SIGMOID_EXPERIMENTS/kitchenD1_sigmoid_pred_gen.sh
# for running inside an interactive GPU session (e.g. GPU-small V100-32GB while
# the ROBO H100s are down). Differences from the sbatch version:
#   - No SBATCH headers; run directly:  bash this_script.sh
#   - BATCH_SIZE defaults to 64 (V100-32GB headroom; H100 scripts use 164).
#   - RSYNC_THREADS defaults to 8 (interactive jobs get ~5 CPUs, not 12).
#   - Predictions are written DIRECTLY to the durable /ocean output dir
#     (no /local ship step): if the session dies, every finished demo is
#     already safe, and re-running resumes via the per-demo skip. Only the
#     INPUT staging uses /local.
# Same checkpoint (epoch 59!), key_suffix, and demo count as the sbatch version.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

GOAL_SOURCE="sigmoid"

# --- GPU sanity check ------------------------------------------------------
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[gpu] ERROR: no GPU visible. Run inside an interactive GPU allocation." >&2
    exit 1
fi
nvidia-smi -L

# --- knobs -----------------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"      # first 100 demos: matches low-level kitchen runs
BATCH_SIZE="${BATCH_SIZE:-64}"     # V100-32GB safe default; raise if memory allows
THREADS="${RSYNC_THREADS:-8}"
echo "[demo_limit] first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- paths -----------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
FINAL_OCEAN_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/ABLATIONS/SIGMOID_EXPERIMENTS_GMM_PREDICTIONS/KITCHEN_D1"

# --- checkpoint (EDIT HERE to change) --------------------------------------
CKPT_PATH="${REPO_DIR}/logs/ABLATION/SIGMOID_EXPERIMENTS/KITCHEN_D1/train_KITCHEN_D1_GOAL_SWAP_SIGMOID_tau1.25_100demo/2026-07-30/20-29-22/checkpoints/periodic-epoch=epoch=59.ckpt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

# --- node-local scratch (inputs only) --------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/KITCHEN_D1_npz"

# --- (1) stage npz source to /local ----------------------------------------
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

# --- (2) generate straight into the durable /ocean tree ---------------------
mkdir -p "${FINAL_OCEAN_DIR}"
cd "${REPO_DIR}"
gen_start=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_pred_to_npz.py \
    --dataset_dir "${DEST_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${FINAL_OCEAN_DIR}" \
    --key_suffix "${GOAL_SOURCE}" \
    --start_demo 0 \
    --max_files "${NUM_DEMOS}" \
    --batch_size "${BATCH_SIZE}"

gen_elapsed=$(( $(date +%s) - gen_start ))
echo "[gen] done in ${gen_elapsed}s. $(find "${FINAL_OCEAN_DIR}" -name '*.npz' | wc -l) npz files now in ${FINAL_OCEAN_DIR}."
