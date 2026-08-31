#!/usr/bin/env bash
# APPROACH 2 eval on Hammer_Cleanup_D1 -- UVD goal-source variant, FROM-GMM.
#
# Same conventions as
#   ../Approach2/eval_approach2_FROMGMM_2d_dit_low_level_hammer_cleanup_d1_DINOV2_c1_0.1_ALL_SEEDS.sh
# (TILING one obs across n_obs_steps, no HL policy -- goal_gripper_pts only
# supervised the aux GMM head during training) but points at the UVD-goal-source
# run synced via sync_best_checkpoints.sh, and evaluates BOTH the best-val_loss
# checkpoint (epoch_20) and the final checkpoint (epoch_99), one seed each.
#
# Per-checkpoint outputs land in:
#   ${SCRIPT_DIR}/APPROACH2_FROMGMM_UVD_2D_DIT_LOW_LEVEL_HAMMER_CLEANUP_50_SAMPLES_D1_DINOV2_c1_0.1_EPOCH<N>/
#
# NOTE: the true best-val_loss checkpoint (epoch 23, nearest saved epoch 20)
# is CORRUPTED on PSC (12MB instead of ~4.1GB, fails to unzip as a torch
# checkpoint) -- several mid-run checkpoints (epoch 15/20/25/30/35/40/45) were
# truncated by some write failure during training. epoch_10.ckpt is the
# closest intact/full-size checkpoint to the true best epoch and is used here
# as the "best" stand-in instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INFERENCE_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
LL_REPO="/home/theyanesh/Pratik_Low_Level/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"
DIT_2D_REPO="${LL_REPO}/diffusion_policy"

ENV_PY="${INFERENCE_ROOT}/.pixi/envs/eval/bin/python"

DATASET_PATH="/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core/hammer_cleanup_d1.hdf5"

# LL: Approach 2 UVD-goal-source run on Hammer_Cleanup_D1, c1 = 0.1.
LL_EXP_DIR="${INFERENCE_ROOT}/theya_approach2_policies/06.48.32_hammercleanup_D1_APPROACH2_uvd_c1_0.1_100demo_dinov2_DIT_hammercleanup_D1_goal_gmm_aux"

ROBOSUITE_ROOT=""

N_EPISODES=50
MAX_STEPS=800
N_OBS_STEPS=2
N_ACTION_STEPS=8
NUM_ENVS="${NUM_ENVS:-8}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-fp32}"
CAMERA_H=256
CAMERA_W=256
SEED=100000

SAVE_VIDEOS=1
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-4}"
VIDEO_FPS=10

OUTPUT_BASE="APPROACH2_FROMGMM_UVD_2D_DIT_LOW_LEVEL_HAMMER_CLEANUP_50_SAMPLES_D1_DINOV2_c1_0.1"
SOURCE_EVAL_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
source "${INFERENCE_ROOT}/shell_scripts/approach2_eval_utils.sh"

for f in "${DATASET_PATH}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${DIT_2D_REPO}/diffusion_policy/policy/flow_matching_dit_goal_gmm_policy.py" \
         "${INFERENCE_ROOT}/eval_smith_utils.py" \
         "${INFERENCE_ROOT}/third_party/robogen/robogen_utils.py" \
         "${INFERENCE_ROOT}/external/robomimic/robomimic/envs/env_robosuite.py" \
         "${LL_REPO}/manipulation/utils.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

cd "${INFERENCE_ROOT}"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}
export PYTHONPATH="${INFERENCE_ROOT}:${LL_REPO}:${PYTHONPATH:-}"

run_one_ckpt() {
  local ckpt="$1" tag="$2" seed="${3:-${SEED}}"
  approach2_prepare_eval_layout \
    "${LL_EXP_DIR}" "${ckpt}" "${SOURCE_EVAL_SCRIPT}" "${OUTPUT_BASE}"
  local OUTPUT_DIR="${APPROACH2_EVAL_ROOT}/${OUTPUT_BASE}_${tag}"

  if [[ ! -e "${LL_EXP_DIR}/checkpoints/${ckpt}" ]]; then
    echo "[ERROR] missing checkpoint: ${LL_EXP_DIR}/checkpoints/${ckpt}" >&2
    exit 1
  fi

  echo
  echo "==========================================================================="
  echo "[${tag}] CKPT=${ckpt}  SEED=${seed}  OUTPUT=${OUTPUT_DIR}"
  echo "==========================================================================="

  export TMPDIR="${TMPDIR:-/tmp}/approach2_fromgmm_uvd_eval_${tag}_$$"
  mkdir -p "${TMPDIR}"

  local VIDEO_FLAG
  if [[ "${SAVE_VIDEOS}" == "0" ]]; then
    VIDEO_FLAG=(--no-save_videos)
  else
    VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
  fi
  local ROBOSUITE_FLAG=()
  if [[ -n "${ROBOSUITE_ROOT}" ]]; then
    ROBOSUITE_FLAG=(--robosuite_root "${ROBOSUITE_ROOT}")
  fi

  "${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_approach2_from_gmm_2d_dit_low_level.py \
      --dataset_path         "${DATASET_PATH}"   \
      --low_level_exp_dir    "${LL_EXP_DIR}"     \
      --low_level_checkpoint "${ckpt}"           \
      --dit_2d_repo          "${DIT_2D_REPO}"    \
      "${ROBOSUITE_FLAG[@]}"                     \
      --n_episodes           "${N_EPISODES}"     \
      --max_steps            "${MAX_STEPS}"      \
      --seed                 "${seed}"           \
      --n_obs_steps          "${N_OBS_STEPS}"    \
      --n_action_steps       "${N_ACTION_STEPS}" \
      --num_envs             "${NUM_ENVS}"       \
      --inference_dtype      "${INFERENCE_DTYPE}" \
      --camera_h             "${CAMERA_H}"       \
      --camera_w             "${CAMERA_W}"       \
      --num_video_episodes   "${NUM_VIDEO_EPISODES}" \
      "${VIDEO_FLAG[@]}"                         \
      --output_dir           "${OUTPUT_DIR}" \
      2>&1 | tee -a "${APPROACH2_EVAL_LOG}"

  approach2_write_combined_summary
}

# run_one_ckpt "epoch_10.ckpt" "EPOCH10"  # already ran, seed 100000
# run_one_ckpt "epoch_99.ckpt" "EPOCH99"  # already ran, seed 100000
# Remaining two seeds of the ALL_SEEDS sweep (100000/150000/250000), EPOCH99 only.
run_one_ckpt "epoch_99.ckpt" "EPOCH99_100000_SEED" "100000"
run_one_ckpt "epoch_99.ckpt" "EPOCH99_150000_SEED" "150000"
run_one_ckpt "epoch_99.ckpt" "EPOCH99_250000_SEED" "250000"

echo
echo "Both checkpoints done. Outputs:"
echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_EPOCH10"
echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_EPOCH99"
echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_EPOCH99_150000_SEED"
echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_EPOCH99_250000_SEED"
