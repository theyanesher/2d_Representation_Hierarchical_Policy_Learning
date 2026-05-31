#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name sweep-dustpan-goal-gripper-100demo-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/RL_TRAINING_DATASET/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/RL_TRAINING_DATASET/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo RL Bench goal_gripper baseline (NATIVE 8-D state/action).
#
# Trains the flow-matching DiT low-level policy on the RL Bench
# sweep_to_dustpan_of_size task, conditioned on the single ground-truth
# goal_gripper_pts (4 keypoints) — NO GMM. Same recipe as the MimicGen
# mug_cleanup goal_gripper baseline, adapted for RL Bench:
#   - 4 cameras (front/left_shoulder/right_shoulder/wrist) at 128x128
#   - native 8-D state/action (pos + quaternion + gripper), action_mode=hybrid_delta
#   - DINOv2 crop 126x126 (9*14; the 224/256 analog for 128px input)
#
# Assumes the .h5 files ALREADY EXIST at SRC_DATA_DIR — the npz->h5 conversion
# is handled separately and will be added later. Stages the first NUM_DEMOS
# files (demo_0.h5 .. demo_(NUM_DEMOS-1).h5) to node-local scratch, then trains.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
# RL Bench npz source (per-frame demo_*/N.npz tree).
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/RL_BENCH_DATASETS/sweep_to_dustpan_of_size/sweep_to_dustpan_of_size"
# Permanent per-demo h5 output on /ocean (persists across runs). Overridable
# via the RL_BENCH_H5_DIR env var.
RL_BENCH_H5_DIR="${RL_BENCH_H5_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/RL_BENCH_DATASETS/Sweep_To_Dustpan_Of_Size}"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- generate npz -> h5 if missing ---------------------------------------
# Mirrors the MimicGen goal_gripper script: consolidate the RL Bench npz tree
# into per-demo h5 ONCE, permanently, into RL_BENCH_H5_DIR on /ocean, then stage
# those h5 to node-local scratch before training. Idempotent: if every demo
# already has its h5 we skip the conversion (and the python startup) entirely.
#
# Uses the same converter as the MimicGen goal_gripper script
# (generate_non_gmm_goals_for_low_level.py) via its --rl_bench branch, which
# reads RL Bench npz (front/left_shoulder/right_shoulder/wrist cameras, native
# 8-D state/action) and writes the same GROOT h5 schema with 4 cameras
# (cam0=front, cam1=left_shoulder, cam2=right_shoulder, cam3=wrist). Per-demo
# idempotent (skips demos whose h5 already exists).
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${RL_BENCH_H5_DIR}" ]; then
    existing_h5_count=$(find "${RL_BENCH_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${RL_BENCH_H5_DIR}: ${existing_h5_count}"

if [ "${src_demo_count}" -gt 0 ] && [ "${existing_h5_count}" -ge "${src_demo_count}" ]; then
    echo "[gen] all ${src_demo_count} demos already converted → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (fills only missing files)"
    mkdir -p "${RL_BENCH_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --rl_bench \
            --no_gmm_output_dir "${RL_BENCH_H5_DIR}"
    )
    echo "[gen] done. *.h5 count now: $(find "${RL_BENCH_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l)"
fi

# Stage from the permanent h5 dir.
SRC_DATA_DIR="${RL_BENCH_H5_DIR}"

# --- node-local scratch --------------------------------------------------
# Always prefer the per-job isolated subdir (/local/slurm-<jobid>/local/) so
# SLURM auto-cleans on job end and concurrent jobs never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Sweep_To_Dustpan_Low_Level_${NUM_DEMOS}demo"

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

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=RL_Bench_Tasks/sweep_to_dustpan_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    "policy.crop_shape=[126,126]" \
    logging.project=rl_bench_tasks \
    logging.name=sweep_to_dustpan_goal_gripper_${NUM_DEMOS}demo_dinov2_DIT \
    name=sweep_to_dustpan_goal_gripper_${NUM_DEMOS}demo_dinov2_DIT \
    dataloader.batch_size=64 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5
