#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH -p GPU-shared         # V100 partition — ROBO H100s are down (2026-08-02)
                              # NOTE: no --ntasks-per-node here — GPU-shared
                              # rejects it ("Requested node configuration is
                              # not available"); CPUs come bundled 5-per-GPU.
#SBATCH --gpus=v100-32:1      # 1x V100-32GB (32GB VRAM: fine for top-6, NOT top2500)
#SBATCH -t 48:00:00           # ~94k steps (943/epoch x 100) AND 2 cameras (vs
                              # PushT's 1) -> ~2-2.5x the PushT V100 wall time,
                              # est. 20-35h. If it times out anyway, relaunch
                              # with RESUME_CKPT=...
#SBATCH --job-name kitchen-d1-wca-noswap-gtmix-v100
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# HIGH-LEVEL ABLATION low-level run on KITCHEN_D1: NO-GOAL-SWAP predictions.
#
# Same recipe as the kitchen WCA baseline (task kitchen_D1_gmm_goal_gt_mix,
# first 100 demos, gt_mix_p=0.5 with GT goals from the h5's own
# goal_gripper_pts — the STANDARD goals the ablation high-level was trained
# on — gmm_top_k=6, 100 epochs) EXCEPT:
#   1. The predicted GMM comes from the NO-GOAL-SWAP ablation high-level
#      (trained without transition_label_swap / weighted sampler) via the
#      parallel npz tree, using the same gmm_pred_npz_dir/key_suffix override
#      mechanism as the coffee RDP gt-mix run. The h5's embedded GMM keys are
#      never loaded.
#   2. Runs on a GPU-shared V100-32 (ROBO down), so BATCH_SIZE defaults to 64
#      (DINOv2+DiT at bs128 will not fit 32GB) and dataloader workers drop
#      to 4 (GPU-shared gives 5 CPUs/GPU).
#
# The npz prediction tree must be fully generated first
# (ROBO_GMM_DATASET_GEN_SCRIPT/ABLATION/...) — staging fails fast otherwise.
#
# Overrides at submission time:
#   GT_MIX_P=0.3 sbatch this_script.sh
#   BATCH_SIZE=128 sbatch this_script.sh          # only on H100
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

SUFFIX="noswap"

# --- knobs ----------------------------------------------------------------
GT_MIX_P="${GT_MIX_P:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_DEMOS="${NUM_DEMOS:-100}"  # first 100 demos: matches all kitchen low-level runs
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth goals per sample)"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

RUN_NAME="groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Kitchen_D1_NOSWAP_GTMIX_p${GT_MIX_P}_top6_bs${BATCH_SIZE}"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="${GMM_H5_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1}"
PRED_SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/ABLATIONS/NO_GOAL_SWAPPING_EXPERIMENTS_GMM_PREDICTIONS/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter).
# num_epochs is an ABSOLUTE target: a resumed run continues from the restored
# epoch and still stops at 100. Default empty (train from scratch):
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_Low_Level_${NUM_DEMOS}demo"
DEST_PRED_DIR="${SCRATCH_ROOT}/KITCHEN_D1_GMM_PRED_${SUFFIX}_${NUM_DEMOS}demo"

# --- stage dataset (h5 files + prediction npz tree) ------------------------
THREADS="${RSYNC_THREADS:-8}"

echo "[stage] h5 source  : ${SRC_DATA_DIR}"
echo "[stage] pred source: ${PRED_SRC_DIR}"
mkdir -p "${DEST_DATA_DIR}" "${DEST_PRED_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR PRED_SRC_DIR DEST_PRED_DIR

# h5 episode files.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

# Ablation prediction npz tree (val + any predicted-GMM samples).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${PRED_SRC_DIR}/$1" "${DEST_PRED_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
pred_count=$(
    find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} h5, ${pred_count} pred dirs."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ] || [ "${pred_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} of each (h5/pred); got ${staged_count}/${pred_count}." >&2
    echo "[stage] Is the ${SUFFIX} prediction generation complete in ${PRED_SRC_DIR}?" >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Build hydra resume overrides only if a checkpoint was requested. The '+' on
# resume_ckpt_path is required because that key isn't in the base config.
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    echo "[resume] resuming training from ${RESUME_CKPT}"
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    RESUME_ARGS=(training.resume=true "+training.resume_ckpt_path=${RESUME_CKPT}")
else
    echo "[resume] RESUME_CKPT empty -> training from scratch"
fi

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/kitchen_D1_gmm_goal_gt_mix \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    +task.dataset.gmm_pred_npz_dir="${DEST_PRED_DIR}" \
    +task.dataset.gmm_pred_key_suffix="${SUFFIX}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name="${RUN_NAME}" \
    name="${RUN_NAME}" \
    training.checkpoint_every=10 \
    training.num_epochs=100 \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
