#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 24:00:00 # 24-hour budget
#SBATCH --job-name franka-push-block-goal-gripper-dinov2
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# Non-GMM goal_gripper baseline on the real-world Franka push-block task
# (tag: franka_push_block). Push-block-to-marked-orientation, PushT-style,
# but collected on a real Franka with the standard MimicGen per-frame npz
# schema (point_cloud, gripper_pcd, goal_gripper_pcd, rgb/depth_agentview,
# rgb/depth_wrist, state(10), action(10)) — same layout as HAMMER_CLEANUP_D1,
# NOT the PushT-specific 2-D npz schema, so we do NOT pass --push_t to the
# converter.
#
# Train flow-matching DiT low-level policy conditioned on the single
# ground-truth goal_gripper_pts (4 keypoints) — NO GMM distribution.
#
# Source on /jet is the per-frame demo_N/ npz tree, so we first consolidate
# it into h5 (idempotent — skips already-converted demos). Demo indices in
# the source tree are NOT guaranteed contiguous from 0 (data may still be
# transferring in), so unlike the fixed-NUM_DEMOS MimicGen scripts, this uses
# ALL demo_*/ dirs present at run time (same recipe as pusht_goal_gripper.sh)
# rather than assuming demo_0..demo_(N-1) exist. Re-run this script later
# (h5 generation + staging are both idempotent / additive) once more demos
# have landed to pick them up.

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/jet/home/eswaramo/data/D2/franka_push_block_mimicgen_npz"
NO_GMM_H5_DIR="/jet/home/eswaramo/data/D2/NO_GMM_preds/franka_push_block_mimicgen"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter)
# from this checkpoint. The flow-matching DiT workspace resumes via
# training.resume=true + training.resume_ckpt_path=<ckpt> (load_checkpoint).
# Default is empty (train from scratch). Override at submission time:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# --- generate npz -> h5 if missing ---------------------------------------
# Conversion is per-demo idempotent (skips demo dirs whose h5 already
# exists). We (re)run it every time so newly-arrived demos get picked up;
# already-converted demos are skipped fast.
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

# --- node-local scratch --------------------------------------------------
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
SRC_DATA_DIR="${NO_GMM_H5_DIR}"
DEST_DATA_DIR="${SCRATCH_ROOT}/Franka_Push_Block_Low_Level"

# --- stage dataset (all h5 files present) ---------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
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

find "${SRC_DATA_DIR}" -mindepth 1 -maxdepth 1 -name '*.h5' -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -eq 0 ]; then
    echo "[stage] ERROR: no h5 files staged." >&2
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
    task=MimicGen_Tasks/franka_push_block_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=franka_push_block_goal_gripper_${staged_count}demo_dinov2_DIT \
    name=franka_push_block_goal_gripper_${staged_count}demo_dinov2_DIT \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
