#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT GMM-LL, HAMMER_CLEANUP_D1, AWE-subgoal 100-demo run.
# HL/LL checkpoints come from logs/train_HAMMER_CLEANUP_D1_AWE_SUBGOALS_100demo/.
# lfd3d and the WCA-capable diffusion_policy come from the theya_high_level /
# theya_low_level worktrees (this branch, MimicGen_DataGen_and_Infer, has neither).
# Uses the `eval` pixi environment (pyopengl>=3.1.10, no pyrender) -- the
# `default` env's pyopengl==3.1.0 can't satisfy mujoco's EGLDeviceEXT import.

set -euo pipefail

REPO_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
ENV_PY="${REPO_ROOT}/.pixi/envs/eval/bin/python"

LFD3D_REPO="/home/theyanesh/worktrees/theya_high_level_lfd3d"
DIT_2D_REPO="/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference/diffusion_policy"

DATASET_PATH="/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core/hammer_cleanup_d1.hdf5"

HL_CKPT="${REPO_ROOT}/logs/train_HAMMER_CLEANUP_D1_AWE_SUBGOALS_100demo/high_level/2026-08-10/13-33-43/checkpoints/epoch=89-step=14310-val/rmse_and_std_combi=0.030.ckpt"

LL_EXP_DIR="${REPO_ROOT}/logs/train_HAMMER_CLEANUP_D1_AWE_SUBGOALS_100demo/low_level"
LL_CKPT="latest.ckpt"

N_EPISODES=50
MAX_STEPS=800
SEED=100000
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256
HL_IN_CHANNELS=4   # 4 = xyz + 1 mask channel (use_rgb=False)

OUTPUT_DIR="logs/eval_GMM_HL_AWE_LL_HAMMER_CLEANUP_D1_100demo"

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
