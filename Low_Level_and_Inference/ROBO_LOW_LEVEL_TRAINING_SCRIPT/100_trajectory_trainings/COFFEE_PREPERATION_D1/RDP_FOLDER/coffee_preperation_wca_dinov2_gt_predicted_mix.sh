#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name coffee-prep-d1-wca-100demo-dinov2-rdp-gtmix
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/RDP_FOLDER/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/RDP_FOLDER/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# RDP GT/predicted goal-MIX run on COFFEE_PREPERATION_D1: per TRAIN sample,
# Bernoulli(gt_mix_p) picks the ground-truth RDP goal set; otherwise the RDP
# high-level prediction:
#   cruise frame      -> 1 mode  = present RDP GT goal,              weight 1.0
#   transition frame  -> 2 modes = [present, neighbor] RDP GT goals, triangular
#                        weights (50/50 at the RDP keypoint, tapering over +-5)
#
# RDP twist vs ../coffee_preperation_wca_dinov2_gt_predicted_mix.sh: the GT
# goals come from the EXTRA_KEYPOINTS npz tree (goal_gripper_pcd_rdp) so
# transition windows follow the RDP keypoint schedule, and VALIDATION (which
# always uses the predicted GMM) is served from the RDP prediction npz tree.
# The h5 files are read-only; their default gmm/goal keys are never loaded.
#
# GT_MIX_P defaults to 0.5. Override: GT_MIX_P=0.3 sbatch this_script.sh
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- gt-mix probability (overridable) -------------------------------------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth RDP goals per sample)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1"
PRED_SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1_GMM_PRED"
GT_SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Low_Level_${NUM_DEMOS}demo"
DEST_PRED_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_GMM_PRED_${NUM_DEMOS}demo"
DEST_GT_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_rdp_goals_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS entries per tree) ----------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] h5 source  : ${SRC_DATA_DIR}"
echo "[stage] pred source: ${PRED_SRC_DIR}"
echo "[stage] gt source  : ${GT_SRC_DIR}"
mkdir -p "${DEST_DATA_DIR}" "${DEST_PRED_DIR}" "${DEST_GT_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR PRED_SRC_DIR DEST_PRED_DIR GT_SRC_DIR DEST_GT_DIR

# h5 episode files.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

# RDP prediction npz tree (val + any predicted-GMM samples).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${PRED_SRC_DIR}/$1" "${DEST_PRED_DIR}/"' _ {}

# RDP GT keypoint npz tree (GT modes for the gt-mix precompute).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${GT_SRC_DIR}/$1" "${DEST_GT_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
pred_count=$(
    find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
gt_count=$(
    find "${DEST_GT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} h5, ${pred_count} pred dirs, ${gt_count} gt dirs."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ] || [ "${pred_count}" -ne "${NUM_DEMOS}" ] || [ "${gt_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} of each (h5/pred/gt); got ${staged_count}/${pred_count}/${gt_count}." >&2
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
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_gmm_goal_gt_mix \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    +task.dataset.gmm_pred_npz_dir="${DEST_PRED_DIR}" \
    +task.dataset.gmm_pred_key_suffix=rdp \
    +task.dataset.gt_goal_npz_dir="${DEST_GT_DIR}" \
    +task.dataset.gt_goal_npz_key=goal_gripper_pcd_rdp \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_RDP_GTMIX_p${GT_MIX_P} \
    name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_RDP_GTMIX_p${GT_MIX_P} \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16
