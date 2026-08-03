#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name hammer-cleanup-d1-goal-gripper-100demo-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo version of the non-GMM goal_gripper baseline, on HAMMER_CLEANUP_D1.
#
# Train flow-matching DiT low-level policy on HAMMER_CLEANUP_D1 conditioned on
# the single ground-truth goal_gripper_pts (4 keypoints) — NO GMM distribution.
# Uses the FIRST NUM_DEMOS demos from the NO_GMM h5 dataset.
#
# Source on /ocean is the GROOT-style demo_N/ npz tree, so we first consolidate
# it into h5 (idempotent — skips already-converted demos). If the NO_GMM h5 dir
# already has at least NUM_DEMOS converted files we skip the python startup
# entirely. Then we stage only the first NUM_DEMOS h5 files to node-local
# scratch before training.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/HAMMER_CLEANUP_D1"
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/HAMMER_CLEANUP_D1"
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
# doesn't already have at least NUM_DEMOS files (demo_0.h5 .. demo_(N-1).h5
# need to exist for the staging step to succeed). --max_files limits the
# conversion to the first NUM_DEMOS demos (numeric order), so the first run
# doesn't convert all 909 demos just to train on 100.
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"

if [ "${existing_h5_count}" -ge "${NUM_DEMOS}" ]; then
    echo "[gen] already have ${existing_h5_count} h5 files (≥ NUM_DEMOS=${NUM_DEMOS}) → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (need at least NUM_DEMOS=${NUM_DEMOS}; have ${existing_h5_count})"
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
            --no_gmm_output_dir "${NO_GMM_H5_DIR}" \
            --max_files "${NUM_DEMOS}"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Hammer_Cleanup_D1_Low_Level_${NUM_DEMOS}demo"

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
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/hammercleanup_D1_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=hammercleanup_D1_goal_gripper_${NUM_DEMOS}demo_dinov2_DIT \
    name=hammercleanup_D1_goal_gripper_${NUM_DEMOS}demo_dinov2_DIT \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
