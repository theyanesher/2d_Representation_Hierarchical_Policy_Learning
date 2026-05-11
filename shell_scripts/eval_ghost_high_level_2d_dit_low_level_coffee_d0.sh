#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT LL on a MimicGen task.
#
# Disentangled from eval_2d_dit.sh — does NOT depend on SMITH_HL_REPO or any
# SigLIP / cat_idx machinery. lfd3d is read-only here; we only import from it.
#
# Heavy training-time deps in the lfd3d package (pytorch_lightning, pytorch3d,
# diffusers, wandb, trimesh, transformers) are stubbed in sys.modules by the
# Python script if missing — they're never reached on the inference path.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths — adjust if any of these dirs move.
# --------------------------------------------------------------------------- #
PROJ_ROOT="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT"

SMITH_MIMICGEN="${PROJ_ROOT}/SMITH_MimicGen/SMITH_on_mimicgen"
LFD3D_REPO="${PROJ_ROOT}/lfd3d/lfd3d"
DIT_2D_REPO="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/diffusion_policy"

# Use the Yufei pixi env's interpreter directly (PyOpenGL 3.1.10 needed for
# mujoco's EGLDeviceEXT; SMITH's own .pixi env has 3.1.0 which is too old).
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/ORIGINAL_DATASET/coffee_d0.hdf5"
HL_CKPT="${LFD3D_REPO}/logs/train_coffeeTask/2026-04-30/19-06-32_MAIN_COFFEE_TASK_PREP/checkpoints/epoch=79-step=100000-val/rmse_and_std_combi=0.016.ckpt"
LL_EXP_DIR="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.04.29/08.39.57_coffee_goal_gripper_DIT_coffee_goal_gripper"
LL_CKPT="epoch_60.ckpt"

# Optional .npy with a (1152,) text embedding for the HL FiLM block.
# Leave empty to use zeros (matches the coffee ckpt's training).
TEXT_EMBED_CACHE=""

# SMITH's external/mimicgen/offcial_robosuite/ is incomplete (no robosuite.models).
# Use the full robosuite checkout that ships under MimicGen/.
ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# --------------------------------------------------------------------------- #
# Eval knobs
# --------------------------------------------------------------------------- #
N_EPISODES=50
MAX_STEPS=400
SEED=100000
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256
HL_IN_CHANNELS=4    # 4 = xyz + 1 mask channel (use_rgb=False, matches the coffee ckpt)
HL_ARGMAX_WEIGHT=1  # 1 = argmax over softmaxed scene weights; 0 = multinomial

# Video saving for every episode. Lands at <OUTPUT_DIR>/media/.
SAVE_VIDEOS=1
VIDEO_FPS=10

# Where to dump args.json / results.jsonl / summary.json.
# Leave empty ("") to let the Python script auto-generate
#   outputs_eval_ghost/<HL_ckpt_stem>__<LL_dir>_<ckpt>/<timestamp>/
OUTPUT_DIR="GHOST_HIGH_LEVEL_2D_DIT_LOW_LEVEL_COFFEE_50_SAMPLES_D0"

# --------------------------------------------------------------------------- #
# Sanity checks before launching
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" "${HL_CKPT}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${LFD3D_REPO}/src/lfd3d/models/articubot.py" \
         "${DIT_2D_REPO}/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py" \
         "${ROBOSUITE_ROOT}/robosuite/models/__init__.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

if [[ -n "${TEXT_EMBED_CACHE}" && ! -e "${TEXT_EMBED_CACHE}" ]]; then
  echo "[ERROR] TEXT_EMBED_CACHE set but missing: ${TEXT_EMBED_CACHE}" >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
cd "${SMITH_MIMICGEN}"

# Same env-var posture as eval_2d_dit.sh — see that script's comments for the why.
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}

# Cosmetic export — the Python script prepends these itself, but non-pixi runs need it.
export PYTHONPATH="${LFD3D_REPO}/src:${DIT_2D_REPO}:${SMITH_MIMICGEN}:${PYTHONPATH:-}"

OUTPUT_DIR_FLAG=()
if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_FLAG=(--output_dir "${OUTPUT_DIR}")
fi

if [[ "${SAVE_VIDEOS}" == "0" ]]; then
  VIDEO_FLAG=(--no-save-videos)
else
  VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
fi

TEXT_EMBED_FLAG=()
if [[ -n "${TEXT_EMBED_CACHE}" ]]; then
  TEXT_EMBED_FLAG=(--text_embed_cache "${TEXT_EMBED_CACHE}")
fi

"${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_ghost_high_level_2d_dit_low_level.py \
    --dataset_path         "${DATASET_PATH}"  \
    --high_level_ckpt      "${HL_CKPT}"       \
    --low_level_exp_dir    "${LL_EXP_DIR}"    \
    --low_level_checkpoint "${LL_CKPT}"       \
    --lfd3d_repo           "${LFD3D_REPO}"    \
    --dit_2d_repo          "${DIT_2D_REPO}"   \
    --robosuite_root       "${ROBOSUITE_ROOT}" \
    --hl_in_channels       "${HL_IN_CHANNELS}" \
    --hl_argmax_weight     "${HL_ARGMAX_WEIGHT}" \
    --n_episodes           "${N_EPISODES}"    \
    --max_steps            "${MAX_STEPS}"     \
    --seed                 "${SEED}"          \
    --n_obs_steps          "${N_OBS_STEPS}"   \
    --n_action_steps       "${N_ACTION_STEPS}" \
    --camera_h             "${CAMERA_H}"      \
    --camera_w             "${CAMERA_W}"      \
    "${TEXT_EMBED_FLAG[@]}"                   \
    "${VIDEO_FLAG[@]}"                        \
    "${OUTPUT_DIR_FLAG[@]}"
