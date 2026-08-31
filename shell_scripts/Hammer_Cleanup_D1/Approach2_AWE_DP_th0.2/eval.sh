#!/usr/bin/env bash
# APPROACH 2 eval -- Hammer_Cleanup_D1, mix_heuristics goal-source, FROM-GMM.
#
# Same conventions as
#   ../Approach2/eval_approach2_FROMGMM_2d_dit_low_level_hammer_cleanup_d1_DINOV2_c1_0.1_ALL_SEEDS.sh
# (TILING one obs across n_obs_steps, no HL policy -- goal_gripper_pts only
# supervised the aux GMM head during training) but points at the
# mix_heuristics-goal-source run (gripper_heuristic + orientation_heuristic
# mixed, see generate_subgoals_nonvlm_D1.sh) synced via sync_best_checkpoints.sh.
# Evaluates ONLY the final checkpoint (epoch_99), across all seeds in SEEDS
# below (100000 / 150000 / 250000 by default).
#
# All task/variant/checkpoint specifics are in the CONFIG block below --
# nothing else in this file should need editing to adopt it for another task
# or another goal-source variant. To adapt:
#   - point DATASET_PATH/LL_RUN_NAME at the new task's hdf5 and synced LL run
#     (sync it first with ./shell_scripts/sync_best_checkpoints.sh)
#   - update TASK_LABEL/VARIANT_LABEL (used only for OUTPUT_BASE naming)
#   - adjust SEEDS/CKPT_NAME/CKPT_TAG if evaluating a different epoch or seed set

set -euo pipefail

# =============================================================================
# CONFIG -- edit this block to adapt to a different task / variant / checkpoint.
# =============================================================================
TASK_LABEL="HAMMER_CLEANUP"        # only used for OUTPUT_BASE naming
VARIANT_LABEL="awe-dp-th0.2"     # only used for OUTPUT_BASE naming
DATASET_PATH="/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core/hammer_cleanup_d1.hdf5"

# Synced via: ./shell_scripts/sync_best_checkpoints.sh '<psc_run_dir_glob>'
# (directory name under theya_approach2_policies/, not a full path)
LL_RUN_NAME="00.01.21_hammercleanup_D1_APPROACH2_awe_dp_c1_0.1_100demo_dinov2_DIT_hammercleanup_D1_goal_gmm_aux"

CKPT_NAME="epoch_99.ckpt"
CKPT_TAG="EPOCH99"                 # only used for OUTPUT_BASE naming
SEEDS=(100000 150000 250000)

N_EPISODES=50
MAX_STEPS=800
N_OBS_STEPS=2
N_ACTION_STEPS=8
NUM_ENVS="${NUM_ENVS:-8}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-fp32}"
CAMERA_H=256
CAMERA_W=256

SAVE_VIDEOS=1
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-4}"
VIDEO_FPS=10

INFERENCE_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
LL_REPO="/home/theyanesh/Pratik_Low_Level/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"
DIT_2D_REPO="${LL_REPO}/diffusion_policy"
ROBOSUITE_ROOT=""
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${INFERENCE_ROOT}/.pixi/envs/eval/bin/python"
LL_EXP_DIR="${INFERENCE_ROOT}/theya_approach2_policies/${LL_RUN_NAME}"
OUTPUT_BASE="APPROACH2_FROMGMM_${VARIANT_LABEL}_2D_DIT_LOW_LEVEL_${TASK_LABEL}_50_SAMPLES_D1_DINOV2_c1_0.1"

# Keep this launcher, its logs, and results beside the policy, grouped by epoch.
source "${INFERENCE_ROOT}/shell_scripts/approach2_eval_utils.sh"
approach2_prepare_eval_layout "${LL_EXP_DIR}" "${CKPT_NAME}" "${BASH_SOURCE[0]}" "${OUTPUT_BASE}"
approach2_start_eval_logging

for f in "${DATASET_PATH}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${CKPT_NAME}" \
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

run_one_seed() {
  local seed="$1"
  local tag="${CKPT_TAG}_${seed}_SEED"
  local OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${tag}"

  echo
  echo "==========================================================================="
  echo "[${tag}] CKPT=${CKPT_NAME}  SEED=${seed}  OUTPUT=${OUTPUT_DIR}"
  echo "==========================================================================="

  export TMPDIR="${TMPDIR:-/tmp}/approach2_fromgmm_${VARIANT_LABEL,,}_eval_${tag}_$$"
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
      --low_level_checkpoint "${CKPT_NAME}"      \
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
      --output_dir           "${OUTPUT_DIR}"
}

for seed in "${SEEDS[@]}"; do
  run_one_seed "${seed}"
done

echo
echo "All ${#SEEDS[@]} seed(s) done. Outputs:"
for seed in "${SEEDS[@]}"; do
  echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_${CKPT_TAG}_${seed}_SEED"
done

approach2_write_combined_summary
