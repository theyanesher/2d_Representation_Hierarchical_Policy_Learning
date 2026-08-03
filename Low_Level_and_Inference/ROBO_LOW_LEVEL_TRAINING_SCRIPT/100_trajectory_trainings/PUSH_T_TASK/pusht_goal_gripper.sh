#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name push-t-task-goal-gripper-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Non-GMM goal_gripper baseline on PushT_Task (tag: push_t_task).
#
# Train flow-matching DiT low-level policy on PushT conditioned on the single
# ground-truth goal_gripper_pts (4 keypoints) — NO GMM distribution.
# Uses ALL demos in the PushT npz tree (206 as of writing).
#
# PushT differences vs the MimicGen tasks (see the task yaml for details):
#   - single agentview camera (cam0 only), no wrist, no depth, no K/T
#   - native 2-D state (pusher xy) and 2-D ABSOLUTE action (target xy for the
#     env's P-controller), written verbatim to the h5 and read as-is by the
#     trainer (action_mode hybrid_delta) — same recipe as the RL Bench tasks.
#
# Source on /ocean is the per-frame demo_N/ npz tree, so we first consolidate
# it into h5 via generate_non_gmm_goals_for_low_level.py --no_gmm --push_t
# (idempotent — skips already-converted demos). If the NO_GMM h5 dir already
# has one h5 per source demo we skip the python startup entirely. Then we
# stage the h5 files to node-local scratch before training.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/PushT_Task"
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/PUSH_T_TASK"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter)
# from this checkpoint. The flow-matching DiT workspace resumes via
# training.resume=true + training.resume_ckpt_path=<ckpt> (load_checkpoint).
# Default is empty (train from scratch). Override at submission time:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# --- generate npz -> h5 if missing ---------------------------------------
# Conversion is per-demo idempotent. We trigger it only if the NO_GMM h5 dir
# doesn't already have one h5 per source demo dir.
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"

if [ "${existing_h5_count}" -ge "${src_demo_count}" ]; then
    echo "[gen] already have ${existing_h5_count} h5 files (≥ ${src_demo_count} src demos) → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (${existing_h5_count}/${src_demo_count} present)"
    mkdir -p "${NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --push_t \
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
DEST_DATA_DIR="${SCRATCH_ROOT}/PushT_Task_Low_Level"

# --- stage dataset (all h5 files) ----------------------------------------
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
if [ "${staged_count}" -ne "${src_demo_count}" ]; then
    echo "[stage] ERROR: expected ${src_demo_count} h5 files staged, got ${staged_count}." >&2
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
    task=MimicGen_Tasks/push_t_task_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=push_t_task_goal_gripper_alldemo_dinov2_DIT \
    name=push_t_task_goal_gripper_alldemo_dinov2_DIT \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    training.num_epochs=200 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
