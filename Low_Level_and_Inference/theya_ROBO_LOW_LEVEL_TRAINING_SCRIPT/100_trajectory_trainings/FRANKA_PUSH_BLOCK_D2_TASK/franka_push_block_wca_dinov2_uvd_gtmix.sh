#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name franka-push-block-UVD-wca-dinov2-gtmix
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/franka-push-block-UVD-wca-dinov2-gtmix_job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/franka-push-block-UVD-wca-dinov2-gtmix_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# UVD GT/predicted goal-MIX run on the real-world Franka push-block task, per
# TRAIN sample: Bernoulli(gt_mix_p) picks the ground-truth UVD goal set;
# otherwise the UVD high-level prediction:
#   cruise frame      -> 1 mode  = present UVD GT goal,              weight 1.0
#   transition frame  -> 2 modes = [present, neighbor] UVD GT goals, triangular
#                        weights (50/50 at the UVD keypoint, tapering over +-5)
#
# Same recipe as coffee_preperation_wca_dinov2_gt_predicted_mix.sh (the RDP
# variant): the GMM predictions and GT goals are read LAZILY from parallel
# per-frame npz trees at train time, NOT baked into a duplicate h5 —
#   - base h5     : plain NO_GMM dataset (obs/goal_gripper_pts, images,
#                   state, action — same h5 the goal_gripper baseline script
#                   generates/uses), read via task.dataset.data_dir.
#   - GMM pred    : EXTRA_KEYPOINTS/FRANKA_PUSH_BLOCK_UVD_GMM_PRED/demo_N/t.npz
#                   (gmm_all_goals_uvd / gmm_all_weights_uvd), produced by
#                   theya_ROBO_GMM_DATASET_GEN_SCRIPT/UVD_DATAGEN/frankaPushBlock.sh
#                   in the high-level repo — run that FIRST.
#   - GT goals    : EXTRA_KEYPOINTS/FRANKA_PUSH_BLOCK/demo_N/t.npz
#                   (goal_gripper_pcd_uvd) — already complete (all 50 demos,
#                   extracted from EXTRA_KEYPOINTS_uvd.tar), no generation
#                   needed.
# The h5 files are read-only; their default gmm/goal keys are never loaded.
#
# GT_MIX_P defaults to 0.5. Override: GT_MIX_P=0.3 sbatch this_script.sh
#
# Source demo indices are NOT guaranteed contiguous from 0 (real-world data
# trickling in), so — unlike the fixed-NUM_DEMOS MimicGen scripts — this
# stages whatever demos are present in ALL THREE sources (base h5 ∩ GMM pred
# ∩ GT) rather than assuming a fixed range. Rerun as more demos land in any
# source to pick them up.

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

GOAL_SOURCE="uvd"

# --- gt-mix probability (overridable) -------------------------------------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth UVD goals per sample)"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/jet/home/eswaramo/data/D2/franka_push_block_mimicgen_npz"
NO_GMM_H5_DIR="/jet/home/eswaramo/data/D2/NO_GMM_preds/franka_push_block_mimicgen"
PRED_SRC_DIR="${GMM_PRED_DIR:-/jet/home/eswaramo/data/D2/EXTRA_KEYPOINTS/FRANKA_PUSH_BLOCK_UVD_GMM_PRED}"
GT_SRC_DIR="/jet/home/eswaramo/data/D2/EXTRA_KEYPOINTS/FRANKA_PUSH_BLOCK"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

if [ ! -d "${PRED_SRC_DIR}" ] || [ -z "$(find "${PRED_SRC_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -print -quit 2>/dev/null)" ]; then
    echo "[error] no GMM prediction demo dirs found in ${PRED_SRC_DIR}" >&2
    echo "[error] run theya_ROBO_GMM_DATASET_GEN_SCRIPT/UVD_DATAGEN/frankaPushBlock.sh in the high-level repo first." >&2
    exit 1
fi
if [ ! -d "${GT_SRC_DIR}" ]; then
    echo "[error] GT goal tree not found: ${GT_SRC_DIR}" >&2
    exit 1
fi

# --- generate npz -> h5 if missing (base NO_GMM dataset) ------------------
# Same idempotent conversion the goal_gripper baseline script uses. Reruns
# here so newly-arrived demos get picked up; already-converted demos are
# skipped fast.
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"

if [ "${existing_h5_count}" -ge "${src_demo_count}" ] && [ "${src_demo_count}" -gt 0 ]; then
    echo "[gen] already have ${existing_h5_count} h5 files (≥ ${src_demo_count} src demos) → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (${existing_h5_count}/${src_demo_count} present)"
    mkdir -p "${NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${NO_GMM_H5_DIR}"
    )
    echo "[gen] done. *.h5 count now: $(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l)"
fi

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter).
# Default is empty (train from scratch). Override at submission time:
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Franka_Push_Block_Low_Level"
DEST_PRED_DIR="${SCRATCH_ROOT}/Franka_Push_Block_GMM_PRED"
DEST_GT_DIR="${SCRATCH_ROOT}/Franka_Push_Block_uvd_goals"

# --- stage all three sources, restricted to demos present in ALL of them ---
THREADS="${RSYNC_THREADS:-32}"

# Demo stems present in the base h5 dir and the GMM prediction tree. The GT
# tree is already complete (all 50 demos) so it's never the limiting set.
demos_h5=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_pred=$(find "${PRED_SRC_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_pred}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), pred demos: $(echo -n "${demos_pred}" | grep -c . || true), common: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both the base h5 dir and the GMM prediction tree." >&2
    exit 1
fi

echo "[stage] h5 source  : ${NO_GMM_H5_DIR}"
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
export NO_GMM_H5_DIR DEST_DATA_DIR PRED_SRC_DIR DEST_PRED_DIR GT_SRC_DIR DEST_GT_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${NO_GMM_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'
echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${PRED_SRC_DIR}/{}" "${DEST_PRED_DIR}/"'
echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${GT_SRC_DIR}/{}" "${DEST_GT_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
pred_count=$(find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
gt_count=$(find "${DEST_GT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} h5, ${pred_count} pred dirs, ${gt_count} gt dirs."
if [ "${staged_count}" -ne "${n_common}" ] || [ "${pred_count}" -ne "${n_common}" ] || [ "${gt_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} of each (h5/pred/gt); got ${staged_count}/${pred_count}/${gt_count}." >&2
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
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/franka_push_block_gmm_goal_gt_mix \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    +task.dataset.gmm_pred_npz_dir="${DEST_PRED_DIR}" \
    +task.dataset.gmm_pred_key_suffix="${GOAL_SOURCE}" \
    +task.dataset.gt_goal_npz_dir="${DEST_GT_DIR}" \
    +task.dataset.gt_goal_npz_key=goal_gripper_pcd_${GOAL_SOURCE} \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_${staged_count}demo_dinov2_Franka_Push_Block_UVD_GTMIX_p${GT_MIX_P} \
    name=groot_GMM_WCA_${staged_count}demo_dinov2_Franka_Push_Block_UVD_GTMIX_p${GT_MIX_P} \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
