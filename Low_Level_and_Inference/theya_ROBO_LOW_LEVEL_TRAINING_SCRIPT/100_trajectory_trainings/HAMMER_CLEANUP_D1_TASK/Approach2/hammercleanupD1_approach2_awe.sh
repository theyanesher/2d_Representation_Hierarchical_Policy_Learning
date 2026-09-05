#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name hammer-cleanup-d1-approach2-awe
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-approach2-awe_job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-approach2-awe_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on HAMMER_CLEANUP_D1  —  goal_source=awe, c1=0.1
# ===========================================================================
# The low-level policy is NOT given the goal. Instead goal_gripper_pts
# supervises a GMM head that reads the same 3D-grounded visual tokens the DiT
# cross-attends to, so the goal shapes the visual representation rather than
# being an input:
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# This variant supervises the GMM head with the awe keypoint field
# from the EXTRA_KEYPOINTS tree (goal_gripper_pcd_awe) instead of
# the ground-truth goal_gripper_pts — selected via +task.dataset.goal_source.
# c1=0.1 is the established "start here" auxiliary-loss weight for Approach 2
# (see the sibling *_c1_0.1.sh scripts in ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/Approach2/
# for how it was chosen).
#
# Pipeline:
#   1. Stage the first NUM_DEMOS h5 files straight from Pratik's shared,
#      read-only NO_GMM dataset to node-local scratch (falls back to
#      consolidating the raw demo_N/ npz tree into a local h5 copy only if
#      that shared dataset doesn't have enough demos for this task).
#   2. Inject obs/goal_gripper_pts_<source> for every source in
#      EXTRA_GOAL_SOURCES (generate_non_gmm_goals_for_low_level.py
#      --inject_extra_goals) from the EXTRA_KEYPOINTS npz tree into the
#      staged, node-local copy. Only appends small (T,4,3) arrays, so it's
#      cheap to redo every job even though the staged copy is ephemeral.
#   3. Train task=MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux with
#      +task.dataset.goal_source=awe and policy.aux_gmm_loss_weight=0.1.
#
# EXTRA_KEYPOINTS tree only covers demo_0..demo_99 (100 demos) for this task,
# so NUM_DEMOS must stay <= 100 or the injection step will fail loudly.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
C1=0.1

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE}, c1=${C1}, NUM_DEMOS=${NUM_DEMOS} (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/jet/home/eswaramo/data/D2/HAMMER_CLEANUP_D1"
EXTRA_GOALS_DIR="/jet/home/eswaramo/data/D2/EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_bspline_bspline_greville_awe_gripper_heuristic_orientation_heuristic_uvd_mix_bspline_bspline_greville_mix_gripper_heuristic_orientation_heuristic/HAMMER_CLEANUP_D1"
# Pratik's already-converted NO_GMM h5s — read-only, so we stage
# straight from here to node-local scratch and inject the goal-source
# keys into the staged copy (see below). LOCAL_NO_GMM_H5_DIR is only
# used as a fallback if this shared dataset lacks enough demos.
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/HAMMER_CLEANUP_D1"
LOCAL_NO_GMM_H5_DIR="/jet/home/eswaramo/data/D2/NO_GMM_preds/HAMMER_CLEANUP_D1"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

# --- resume from checkpoint ----------------------------------------------
# Empty by default: a fresh goal-source variant should NOT resume from a
# different variant's checkpoint. Override at submission time if resuming a
# previous run OF THIS SAME VARIANT:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# --- resolve NO_GMM h5 source: shared (read-only) or local (generate) ----
shared_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    shared_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] shared *.h5 in ${NO_GMM_H5_DIR}: ${shared_h5_count}"

if [ "${shared_h5_count}" -ge "${NUM_DEMOS}" ]; then
    echo "[gen] shared dataset has enough demos -> staging straight from it"
else
    echo "[gen] shared dataset insufficient (${shared_h5_count} < ${NUM_DEMOS}) -> falling back to local generation"
    src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
    existing_h5_count=0
    if [ -d "${LOCAL_NO_GMM_H5_DIR}" ]; then
        existing_h5_count=$(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
    fi
    echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
    echo "[gen] existing local *.h5 in ${LOCAL_NO_GMM_H5_DIR}: ${existing_h5_count}"
    if [ "${existing_h5_count}" -ge "${NUM_DEMOS}" ]; then
        echo "[gen] already have ${existing_h5_count} local h5 files (>= NUM_DEMOS=${NUM_DEMOS}) -> skipping h5 generation"
    else
        echo "[gen] converting demo_*/ -> demo_*.h5 (need at least NUM_DEMOS=${NUM_DEMOS}; have ${existing_h5_count})"
        mkdir -p "${LOCAL_NO_GMM_H5_DIR}"
        (
            cd "${REPO_DIR}"
            USE_TF=0 \
            GIT_LFS_SKIP_SMUDGE=1 \
            PYTHONNOUSERSITE=1 \
            pixi run python generate_non_gmm_goals_for_low_level.py \
                --dataset_dir "${SRC_NPZ_DIR}" \
                --no_gmm \
                --no_gmm_output_dir "${LOCAL_NO_GMM_H5_DIR}" \
                --max_files "${NUM_DEMOS}"
        )
        echo "[gen] done. *.h5 count now: $(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l)"
    fi
    NO_GMM_H5_DIR="${LOCAL_NO_GMM_H5_DIR}"
fi

# --- node-local scratch --------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
SRC_DATA_DIR="${NO_GMM_H5_DIR}"
DEST_DATA_DIR="${SCRATCH_ROOT}/Hammer_Cleanup_D1_Approach2_${GOAL_SOURCE}_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"
echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] files  : demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5  (${NUM_DEMOS} files)"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

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

# --- inject extra goal keys into the staged, node-local copy -------------
# Injecting here (not into NO_GMM_H5_DIR) because that may be Pratik's
# shared read-only dataset. Only appends small (T,4,3) arrays, so redoing
# this every job (staging is ephemeral) is cheap.
echo "[inject] ensuring extra goal keys exist in staged demos"
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${DEST_DATA_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${EXTRA_GOALS_DIR}" \
        --max_files "${NUM_DEMOS}"
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

RUN_NAME="hammercleanup_D1_APPROACH2_${GOAL_SOURCE}_c1_${C1}_${NUM_DEMOS}demo_dinov2_DIT"

# batch_size=128 matches the goal_gripper baseline. Measured scaling for this
# architecture is ~0.24 GiB/sample over a ~2.75 GiB floor, i.e. ~34 GiB at 128
# -- comfortable on an 80 GB H100.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/jet/home/eswaramo/logs/wandb_cache \
WANDB_DATA_DIR=/jet/home/eswaramo/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
