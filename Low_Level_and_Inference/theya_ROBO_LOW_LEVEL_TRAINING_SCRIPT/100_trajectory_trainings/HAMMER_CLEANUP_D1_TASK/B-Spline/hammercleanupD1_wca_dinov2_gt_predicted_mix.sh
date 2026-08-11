#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 24:00:00 # 24-hour budget
#SBATCH --job-name hammer-cleanup-d1-BSpline-wca-100demo-dinov2-gtmix-goalswap
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-BSpline-wca-100demo-dinov2-gtmix-goalswap_job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-BSpline-wca-100demo-dinov2-gtmix-goalswap_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# 100-demo GT/predicted goal-MIX run on HAMMER_CLEANUP_D1, using the h5 dataset
# whose GMM annotations come from the BSpline-subgoal-trained high-level model
# (HAMMER_CLEANUP_D1_BSpline, trained via
# logs/train_HammerCleanup_D1_BSpline/2026-08-08/18-51-00,
# best-val checkpoint checkpoints/epoch=14-step=6090-val/rmse_and_std_combi=0.034.ckpt,
# GMM dataset already generated locally at
# /data/theya/D1/HAMMER_CLEANUP_D1_BSpline_SUBGOALS_100demo via
# scripts/run_gmm_on_dataset_batch_optimized.py on the theya_high_level pixi env).
#
# Same recipe as the coffee gt_predicted_mix run: the additive task
# `hammercleanup_D1_gmm_goal_gt_mix`, whose dataset subclass
# (LazyArticuBotGtMixDataset) replaces the predicted GMM goals with sparse
# ground-truth goals on a per-item Bernoulli(gt_mix_p); the rest of the time it
# passes the predicted high-level GMM through unchanged. Same data, same
# container, same policy / gmm_top_k=6 — only the goal *contents* are mixed per
# sample. p lives in the config (gt_mix_p=0.5); override here or at launch:
#   GT_MIX_P=0.3 sbatch this_script.sh
#   NUM_DEMOS=200 sbatch this_script.sh   (needs the BSpline GMM dataset
#                                          regenerated with NUM_DEMOS>=200 first)
# To train against a different GMM annotation source instead:
#   GMM_H5_DIR=/data/theya/D1/HAMMER_CLEANUP_D1_BSPLINE_SUBGOALS_100demo sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- gt-mix probability (lives in the config; overridable here) -----------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth goals per sample)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="${GMM_H5_DIR:-/jet/home/eswaramo/data/D2/GMM_preds/BSpline/HAMMER_CLEANUP_D1_BSPLINE_SUBGOALS_100demo}"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Hammer_Cleanup_D1_Bspline_Low_Level_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
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
WANDB_CACHE_DIR=/jet/home/eswaramo/logs/wandb_cache \
WANDB_DATA_DIR=/jet/home/eswaramo/logs/wandb_data \
PYTHONNOUSERSITE=1 \
# PIXI_CACHE_DIR=/home/theyanesh/.cache/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/hammercleanup_D1_gmm_goal_gt_mix \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Hammer_Cleanup_D1_Bspline_GTMIX_p${GT_MIX_P} \
    name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Hammer_Cleanup_D1_Bspline_GTMIX_p${GT_MIX_P} \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
