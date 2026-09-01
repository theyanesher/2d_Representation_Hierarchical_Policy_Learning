#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 24:00:00 # 24-hour budget
#SBATCH --job-name pourbarley-approach2-awe-greedy-th0.35
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pourbarley-approach2-awe-greedy-th0.35_job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pourbarley-approach2-awe-greedy-th0.35_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on the real-world Franka POUR_BARLEY task
#                                     —  goal_source=awe (greedy, err_th=0.35), c1=0.1
# ===========================================================================
# Direct counterpart of FRANKA_PUSH_BLOCK_D2_TASK/Approach2/pushblock_approach2_awe_greedy_th0.1.sh
# for PourBarley_Tasks/pour_barley_goal_gmm_aux. The low-level policy is NOT
# given the goal. Instead goal_gripper_pts supervises a GMM head that reads
# the same 3D-grounded visual tokens the DiT cross-attends to, so the goal
# shapes the visual representation rather than being an input:
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# This variant supervises the GMM head with the awe keypoint field
# (goal_gripper_pcd_awe) sourced from the standalone
# EXTRA_KEYPOINTS_pourbarley_awe-greedy-th0.35 tree — AWE subgoals produced by
# the greedy keypoint-selection method at error threshold 0.35. That tree is
# still being generated (only a subset of demos present as of writing), so
# this script stages the INTERSECTION of demos present in both
# POUR_BARLEY_H5_DIR and EXTRA_GOALS_DIR rather than assuming a fixed
# demo_0..N-1 range — same pattern as the pushblock awe script. Rerun as more
# demos land in the keypoints tree to pick them up.
#
# Unlike the MimicGen Approach2 scripts, this does NOT run
# generate_non_gmm_goals_for_low_level.py --no_gmm here: the pour_barley h5
# tree is produced upstream already cropped/resized to the 256x256 shape
# pour_barley_goal_gmm_aux.yaml expects (agentview/wrist crop + intrinsics
# homography, preprocessing_mode='pour_barley' — see the yaml's header
# comment), so this script only stages and trains.
#
# POUR_BARLEY_H5_DIR / EXTRA_GOALS_DIR default to the paths the data is
# expected to land at. Override at submission time if the final locations
# differ once generation completes:
#   POUR_BARLEY_H5_DIR=/path/to/h5 EXTRA_GOALS_DIR=/path/to/keypoints sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
# SOURCE_TAG only distinguishes run/log/scratch naming from any other
# awe-sourced sibling script for this task — it is NOT passed to hydra
# (goal_source stays "awe" because that's the npz/h5 key name in every tree).
SOURCE_TAG="awe_greedy_th0.35"
C1=0.1

# --- paths -----------------------------------------------------------------
POUR_BARLEY_H5_DIR="${POUR_BARLEY_H5_DIR:-/jet/home/eswaramo/data/D2/pour_barley_h5}"
EXTRA_GOALS_DIR="${EXTRA_GOALS_DIR:-/jet/home/eswaramo/data/D2/EXTRA_KEYPOINTS_pourbarley_awe-greedy-th0.35/pour_barley_npz}"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}"
echo "[config] POUR_BARLEY_H5_DIR=${POUR_BARLEY_H5_DIR}"
echo "[config] EXTRA_GOALS_DIR=${EXTRA_GOALS_DIR}"

if [ ! -d "${POUR_BARLEY_H5_DIR}" ]; then
    echo "[error] POUR_BARLEY_H5_DIR not found: ${POUR_BARLEY_H5_DIR}" >&2
    echo "[error] Has the h5 tree landed yet? Override with POUR_BARLEY_H5_DIR=... if it moved elsewhere." >&2
    exit 1
fi
if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    echo "[error] Override with EXTRA_GOALS_DIR=... if the subgoals tree landed elsewhere." >&2
    exit 1
fi

# --- resume from checkpoint ----------------------------------------------
# Empty by default: a fresh goal-source variant should NOT resume from a
# different variant's checkpoint. Override at submission time if resuming a
# previous run OF THIS SAME VARIANT:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# --- node-local scratch --------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/PourBarley_Approach2_${SOURCE_TAG}"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree --
THREADS="${RSYNC_THREADS:-32}"

demos_h5=$(find "${POUR_BARLEY_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both POUR_BARLEY_H5_DIR and EXTRA_GOALS_DIR." >&2
    exit 1
fi

echo "[stage] source : ${POUR_BARLEY_H5_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export POUR_BARLEY_H5_DIR DEST_DATA_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${POUR_BARLEY_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- inject obs/goal_gripper_pts_awe into the staged, node-local copy ----
# Reads EXTRA_GOALS_DIR directly (small per-frame data, no need to stage it).
# Only appends a small (T,4,3) array, so redoing this every job (staging is
# ephemeral) is cheap.
echo "[inject] ensuring obs/goal_gripper_pts_${GOAL_SOURCE} exists in staged demos (from EXTRA_KEYPOINTS_pourbarley_awe-greedy-th0.35)"
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${DEST_DATA_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${EXTRA_GOALS_DIR}"
)

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

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

RUN_NAME="pourbarley_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT"

# batch_size=128 matches the goal_gripper baseline / other Approach2 scripts.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/jet/home/eswaramo/logs/wandb_cache \
WANDB_DATA_DIR=/jet/home/eswaramo/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=PourBarley_Tasks/pour_barley_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
