#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name kitchen-d1-approach2-rate-ablation-5hz
#SBATCH -o /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-rate-ablation-5hz_job_%j.out
#SBATCH -e /jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-rate-ablation-5hz_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on KITCHEN_D1  —  control-rate ablation, 5 Hz, goal_source=awe (greedy, err_th=0.35, with-gripper), c1=0.1
# ===========================================================================
# Same GMM-as-auxiliary-loss setup as kitchenD1_approach2_awe_greedy_th0.35_nogrip.sh:
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# but trained on the rate-ablated demos under /jet/home/eswaramo/data/rate_ablations,
# which re-record the same KITCHEN_D1 trajectories at a lower control frequency
# (5 Hz here, vs the ~20 Hz baseline used by every other kitchenD1_approach2_*.sh
# sibling script). No model/config changes are needed for this: the h5 schema
# (obs/goal_gripper_pts, present_gripper_pts, point_cloud, images, depth,
# action/delta, action/hybrid) and the EXTRA_KEYPOINTS npz format
# (goal_gripper_pcd_awe, shape (1,4,3), demo_N/t.npz layout) are byte-identical
# to what generate_non_gmm_goals_for_low_level.py --inject_extra_goals already
# expects, and this workspace (train_diffusion_unet_hybrid_workspace.py) has
# its env_runner/rollout-eval path commented out, so horizon/n_obs_steps/
# n_action_steps (frame counts) and env_runner.fps in the task yaml are never
# exercised — training is pure offline supervised learning on the h5 dataset,
# agnostic to the real-world control rate the frames were sampled at.
#
# NOT run at 20 Hz: /jet/home/eswaramo/data/rate_ablations/h5/kitchen_d1_20hz
# has demo_0 T=586 frames, matching the existing NO_GMM baseline's demo_0
# T=585 (verified 2026-08-30) -- i.e. the established kitchenD1_approach2_awe.sh
# and its th0.2/th0.6/th0.35_nogrip siblings already ARE the 20 Hz condition,
# so a dedicated 20 Hz rate-ablation script would be a near-duplicate.
#
# The rate_ablations EXTRA_KEYPOINTS tree is SPARSE like the th0.35_nogrip
# tree: 106/210 demo ids, non-contiguous (e.g. demo_0, demo_3, demo_5, ...,
# demo_209 -- no fixed demo_0..N-1 range), but identical between the h5 dir
# and EXTRA_KEYPOINTS dir for a given rate, and identical across all rates
# (same 106 underlying demos, just resampled) -- verified 2026-08-30. So this
# script first computes the intersection of demos present in both the h5 dir
# and EXTRA_GOALS_DIR (same check as kitchenD1_approach2_awe_greedy_th0.35_nogrip.sh),
# then takes the first NUM_DEMOS of that intersection by ascending numeric
# demo id -- so every rate in the sweep trains on the SAME 100 underlying
# demos (just resampled), matching the 100-demo convention used by every
# other script in this 100_trajectory_trainings/ tree. NUM_DEMOS defaults to
# 100; override at submission time:
#   NUM_DEMOS=50 sbatch this_script.sh
#
# Pipeline:
#   1. Stage the first NUM_DEMOS demos (by numeric id) common to the
#      rate-specific h5 dir and EXTRA_GOALS_DIR to node-local scratch (both
#      already exist on disk -- no raw npz -> h5 conversion needed, unlike
#      the D2-tree baseline scripts).
#   2. Inject obs/goal_gripper_pts_awe (generate_non_gmm_goals_for_low_level.py
#      --inject_extra_goals) from the rate-specific EXTRA_KEYPOINTS npz tree
#      into the staged, node-local copy.
#   3. Train task=MimicGen_Tasks/kitchen_goal_gmm_aux with
#      +task.dataset.goal_source=awe and policy.aux_gmm_loss_weight=0.1.

set -euo pipefail
set -x

export PIXI_HOME="/jet/home/eswaramo/data/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
HZ=5
GOAL_SOURCE="awe"
# SOURCE_TAG only distinguishes run/log/scratch naming from the other
# rate_ablation_*hz siblings and from the baseline (~20Hz) awe scripts -- it
# is NOT passed to hydra (goal_source stays "awe" because that's the npz/h5
# key name in every tree).
SOURCE_TAG="rate_ablation_${HZ}hz_awe_greedy_th0.35_grip"
C1=0.1

# --- paths ---------------------------------------------------------------
NO_GMM_H5_DIR="/jet/home/eswaramo/data/rate_ablations/h5/kitchen_d1_${HZ}hz"
EXTRA_GOALS_DIR="/jet/home/eswaramo/data/rate_ablations/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/kitchen_d1_${HZ}hz"
REPO_DIR="/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/"

echo "[config] hz=${HZ}, goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}"
echo "[config] NO_GMM_H5_DIR=${NO_GMM_H5_DIR}"
echo "[config] EXTRA_GOALS_DIR=${EXTRA_GOALS_DIR}"

if [ ! -d "${NO_GMM_H5_DIR}" ]; then
    echo "[error] NO_GMM_H5_DIR not found: ${NO_GMM_H5_DIR}" >&2
    exit 1
fi
if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi

# --- resume from checkpoint ----------------------------------------------
# Empty by default: a fresh rate variant should NOT resume from a different
# rate's checkpoint. Override at submission time if resuming a previous run
# OF THIS SAME VARIANT:
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_Approach2_${SOURCE_TAG}"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree --
THREADS="${RSYNC_THREADS:-32}"

NUM_DEMOS="${NUM_DEMOS:-100}"

demos_h5=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common_all=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common_all=$(echo -n "${demos_common_all}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common_all}"
if [ "${n_common_all}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both NO_GMM_H5_DIR and EXTRA_GOALS_DIR." >&2
    exit 1
fi
if [ "${n_common_all}" -lt "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: requested NUM_DEMOS=${NUM_DEMOS} but only ${n_common_all} demos are common to both trees." >&2
    exit 1
fi

# demo ids are sparse/non-contiguous -- take the first NUM_DEMOS by ascending
# NUMERIC id (plain `sort` above is lexicographic: demo_10 < demo_2), so the
# same 100 underlying demos are used at every rate in the sweep.
demos_common=$(echo "${demos_common_all}" | sed -E 's/^demo_([0-9]+)$/\1/' | sort -n | sed -E 's/^/demo_/' | head -n "${NUM_DEMOS}")
n_common="${NUM_DEMOS}"
echo "[stage] using first ${NUM_DEMOS} demos (by numeric id) of ${n_common_all} common demos"

echo "[stage] source : ${NO_GMM_H5_DIR}"
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
export NO_GMM_H5_DIR DEST_DATA_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${NO_GMM_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- inject extra goal keys into the staged, node-local copy -------------
# Injecting here (not into NO_GMM_H5_DIR) because that tree may be reused
# read-only by sibling rate scripts. Only appends a small (T,4,3) array, so
# redoing this every job (staging is ephemeral) is cheap.
echo "[inject] ensuring obs/goal_gripper_pts_awe exists in staged demos (from ${SOURCE_TAG})"
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

RUN_NAME="kitchen_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT"

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
    task=MimicGen_Tasks/kitchen_goal_gmm_aux \
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
