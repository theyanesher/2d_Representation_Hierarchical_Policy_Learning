#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name push-t-wca-alpha0.1-ablation
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ALPHA ABLATION on PushT_Task GOAL_SWAP (DINOv2, serial WCA, GT-mix) — ALPHA=0.1 variant.
#
# Same treatment as the KITCHEN_D1 alpha ablation, applied to the PushT recipe.
# Compared to the top-6 GT-mix baseline (pusht_wca_dinov2_gt_predicted_mix.sh),
# this run:
#   1. passes many more GMM candidates to the policy (gmm_top_k=TOP_K, default
#      2500 of the 4500 anchors). On PushT GOAL_SWAP this matters even more than
#      on Kitchen: the predicted GMM is very diffuse — top-6 carries only ~6.7%
#      mean (1.6% min) of the probability mass, so the old baseline discarded
#      ~93% of the posterior on average.
#   2. scales ONLY the WeightedCrossAttention content logits by ALPHA:
#          logits = ALPHA * QK^T/sqrt(d) + log(w_j)
#      Small ALPHA makes the high-level GMM weights dominate candidate ranking.
#      NOTE: because the PushT weights are so flat, the log-weight prior is weak
#      here — expect alpha to bite less than on Kitchen at the same value.
#      No other attention layer (visual CA / self-attn) is touched.
#
# Everything else matches the GT-mix baseline: task push_t_task_gmm_goal_gt_mix
# with gt_mix_p=GT_MIX_P (default 0.5), ALL demos, single agentview camera,
# 200 epochs, checkpoint every 5, optional RESUME_CKPT.
#
# Launch matrix (env overrides):
#   sbatch this_script.sh                       # top-2500, alpha=0.1, gtmix 0.5
#   ALPHA=0.3  sbatch this_script.sh            # top-2500, alpha=0.3
#   USE_ALPHA=false sbatch this_script.sh       # top-2500 CONTROL (original
#                                               #   formula, no alpha) — needed to
#                                               #   separate "more goals" from "alpha"
#   TOP_K=6 USE_ALPHA=false sbatch this_script.sh   # reproduces the old baseline
#   GT_MIX_P=0.3 sbatch this_script.sh          # different GT-mix probability
#
# MEMORY NOTE: goal tokens + per-block WCA K/V activations scale linearly in
# TOP_K * batch_size (~25 GB extra fp32 at TOP_K=2500 / batch 64, ~50 GB at
# batch 128 — borderline on an 80 GB H100 even with PushT's single camera), so
# BATCH_SIZE defaults to 64. If you change batch relative to the batch-128
# baseline, consider re-running the baseline at the matched batch for fairness.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- ablation knobs -------------------------------------------------------
ALPHA="${ALPHA:-0.1}"          # wca_alpha value (used only when USE_ALPHA=true)
USE_ALPHA="${USE_ALPHA:-true}" # false => original WCA formula (alpha off)
TOP_K="${TOP_K:-2500}"         # gmm_top_k candidates passed to the policy
BATCH_SIZE="${BATCH_SIZE:-64}" # see MEMORY NOTE above

# --- gt-mix probability (lives in the config; overridable here) -----------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth goals per sample)"

if [ "${USE_ALPHA}" = "true" ]; then
    ALPHA_TAG="alpha${ALPHA}"
else
    ALPHA_TAG="noalpha"
fi

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="${GMM_H5_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/PUSH_T_TASK_GOAL_SWAP}"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- demo selection (default: ALL h5 files in the source dir) --------------
src_h5_count=$(find "${SRC_DATA_DIR}" -maxdepth 1 -name 'demo_*.h5' 2>/dev/null | wc -l)
if [ "${src_h5_count}" -eq 0 ]; then
    echo "[stage] ERROR: no demo_*.h5 in ${SRC_DATA_DIR}." >&2
    echo "[stage] Run ROBO_GMM_DATASET_GEN_SCRIPT/pushT_GOAL_SWAP.sh first to generate the dataset." >&2
    exit 1
fi
NUM_DEMOS="${NUM_DEMOS:-${src_h5_count}}"
echo "[demo_limit] using ${NUM_DEMOS} of ${src_h5_count} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

RUN_NAME="groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_PushT_Task_GOALSWAP_GTMIX_p${GT_MIX_P}_top${TOP_K}_${ALPHA_TAG}_bs${BATCH_SIZE}"
echo "[ablation] TOP_K=${TOP_K}  USE_ALPHA=${USE_ALPHA}  ALPHA=${ALPHA}  BATCH_SIZE=${BATCH_SIZE}"
echo "[ablation] run name: ${RUN_NAME}"

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter).
# Default is empty (train from scratch). Override at submission time:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide. PSC's SLURM doesn't always pre-create
# that subtree, so we force-create it ourselves when SLURM_JOB_ID is set. Fall
# back to $LOCAL or /tmp only if no per-job dir exists.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/PushT_Task_GoalSwap_Low_Level_${NUM_DEMOS}demo"

# --- stage dataset (NUM_DEMOS files) --------------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] files  : demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5  (${NUM_DEMOS} files)"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Generate exactly the demo filenames we want and feed to xargs.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
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
    task=MimicGen_Tasks/push_t_task_gmm_goal_gt_mix \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.use_alpha="${USE_ALPHA}" \
    policy.wca_alpha="${ALPHA}" \
    policy.gmm_top_k="${TOP_K}" \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name="${RUN_NAME}" \
    name="${RUN_NAME}" \
    training.checkpoint_every=5 \
    training.num_epochs=200 \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
