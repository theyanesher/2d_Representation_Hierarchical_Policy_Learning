#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT GMM-LL, COFFEE_PREPARATION_D1, 100-demo run.
# Usage: ./eval_gmm_high_level_2d_dit_low_level_coffee_preperation_d1_local.sh <awe|bspline> <seed>
#   e.g. ./eval_..._local.sh awe 100000
# HL checkpoint comes from logs/train_COFFEE_PREPERATION_D1_<VARIANT>_subgoals_100demo/.
# LL checkpoint comes from logs/<ts>_groot_GMM_WCA_100demo_dinov2_Coffee_Preperation_D1_<VARIANT>_GTMIX_p0.5_.../.
# lfd3d and the WCA-capable diffusion_policy come from the theya_high_level /
# theya_low_level worktrees (this branch, MimicGen_DataGen_and_Infer, has neither).
# Uses the `eval` pixi environment (pyopengl>=3.1.10, no pyrender) -- the
# `default` env's pyopengl==3.1.0 can't satisfy mujoco's EGLDeviceEXT import.

set -euo pipefail

VARIANT="${1:-bspline}"   # awe | bspline
SEED="${2:-150000}"

REPO_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
ENV_PY="${REPO_ROOT}/.pixi/envs/eval/bin/python"

LFD3D_REPO="/home/theyanesh/worktrees/theya_high_level_lfd3d"
DIT_2D_REPO="/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference/diffusion_policy"

DATASET_PATH="/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core/coffee_preparation_d1.hdf5"

case "${VARIANT}" in
  awe)
    HL_CKPT="${REPO_ROOT}/logs/train_COFFEE_PREPERATION_D1_AWE_subgoals_100demo/checkpoints/epoch=59-step=22500-val/rmse_and_std_combi=0.023.ckpt"
    LL_EXP_DIR="${REPO_ROOT}/logs/19.13.57_groot_GMM_WCA_100demo_dinov2_Coffee_Preperation_D1_AWE_GTMIX_p0.5_coffee_preperation_gmm_goal_gt_mix"
    ;;
  bspline)
    HL_CKPT="${REPO_ROOT}/logs/train_COFFEE_PREPERATION_D1_BSPLINE_subgoals_100demo/checkpoints/epoch=59-step=22500-val/rmse_and_std_combi=0.027.ckpt"
    LL_EXP_DIR="${REPO_ROOT}/logs/19.22.48_groot_GMM_WCA_100demo_dinov2_Coffee_Preperation_D1_BSPLINE_GTMIX_p0.5_coffee_preperation_gmm_goal_gt_mix"
    ;;
  *)
    echo "[ERROR] unknown variant '${VARIANT}', expected 'awe' or 'bspline'" >&2
    exit 1
    ;;
esac
LL_CKPT="latest.ckpt"

N_EPISODES=50
MAX_STEPS=800
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256
HL_IN_CHANNELS=4   # 4 = xyz + 1 mask channel (use_rgb=False)

VARIANT_UPPER="$(echo "${VARIANT}" | tr '[:lower:]' '[:upper:]')"
OUTPUT_DIR="logs/eval_GMM_HL_${VARIANT_UPPER}_LL_COFFEE_PREPERATION_D1_100demo_seed${SEED}"

# --------------------------------------------------------------------------- #
# Sanity checks before launching
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" "${HL_CKPT}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${LFD3D_REPO}/src/lfd3d/models/articubot.py" \
         "${DIT_2D_REPO}/diffusion_policy/model/flow_matching/cross_attention_dit.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}
export PYTHONPATH="${LFD3D_REPO}/src:${DIT_2D_REPO}:${REPO_ROOT}:${PYTHONPATH:-}"
# mujoco_py (legacy binding, used by robomimic's EnvRobosuite) needs the
# standalone MuJoCo 2.1 native lib + nvidia GL libs on LD_LIBRARY_PATH, and
# `patchelf` (from the eval env's conda deps) on PATH to fix up its rpath
# during its one-time Cython extension build.
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export PATH="${REPO_ROOT}/.pixi/envs/eval/bin:${PATH}"

"${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_gmm_high_level_2d_dit_low_level.py \
    --dataset_path         "${DATASET_PATH}"  \
    --high_level_ckpt      "${HL_CKPT}"       \
    --low_level_exp_dir    "${LL_EXP_DIR}"    \
    --low_level_checkpoint "${LL_CKPT}"       \
    --lfd3d_repo           "${LFD3D_REPO}"    \
    --dit_2d_repo          "${DIT_2D_REPO}"   \
    --hl_in_channels       "${HL_IN_CHANNELS}" \
    --n_episodes           "${N_EPISODES}"    \
    --max_steps            "${MAX_STEPS}"     \
    --seed                 "${SEED}"          \
    --n_obs_steps          "${N_OBS_STEPS}"   \
    --n_action_steps       "${N_ACTION_STEPS}" \
    --camera_h             "${CAMERA_H}"      \
    --camera_w             "${CAMERA_W}"      \
    --save_videos --video_fps 10              \
    --output_dir           "${OUTPUT_DIR}"
