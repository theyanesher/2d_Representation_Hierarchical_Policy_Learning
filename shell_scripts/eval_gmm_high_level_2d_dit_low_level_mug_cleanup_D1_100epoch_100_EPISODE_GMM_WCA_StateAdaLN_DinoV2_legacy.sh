#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT GMM-LL on a MimicGen task.
#
# MugCleanup_D1 variant. Sister of eval_gmm_high_level_2d_dit_low_level_hammer_cleanup_d1.sh.
# Same env / same action-conversion path. The LL is the GMM-conditioned variant
# trained with:
#   task=MimicGen_Tasks/mug_cleanup_gmm_goal
#   policy.use_goal_cross_attention=true
#   policy.use_weighted_cross_attention=true
#   policy.gmm_top_k=1024
#
# At inference we feed the FULL N=4500 anchor distribution from the HL into the
# LL; the LL's gmm_top_k is applied internally inside FlowMatchingDiTImagePolicy.
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

DATASET_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/ORIGINAL_DATASET/mug_cleanup_d1.hdf5"

# HL: MULTIMODAL Articubot ckpt trained on Mug_Cleanup_d2.
HL_CKPT="${PROJ_ROOT}/SMITH_High_Level_FineTune/HIGH_LEVEL_POLICIES/Mug_Cleanup_d2/GHOST_High_Level_MULTIMODAL/periodic-epoch=epoch=44.ckpt"

# LL: the MugCleanup_D1 100-demo Resnet run. Hydra exp dir + checkpoint name.
LL_EXP_DIR="${PROJ_ROOT}/SMITH_High_Level_FineTune/LOW_LEVEL_POLICIES/MugCleanup_D1/100_Demos_Model/DinoV2_model_GMM_WCA_State_AdaLN/"
LL_CKPT="epoch_95.ckpt"

# Optional .npy with a (1152,) text embedding for the HL FiLM block.
# Leave empty to use zeros (matches the HL ckpt's training).
TEXT_EMBED_CACHE=""

# SMITH's external/mimicgen/offcial_robosuite/ is incomplete (no robosuite.models).
# Use the full robosuite checkout that ships under MimicGen/.
ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# --------------------------------------------------------------------------- #
# Eval knobs
# --------------------------------------------------------------------------- #
N_EPISODES=50
MAX_STEPS=800
SEED=250000
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256
HL_IN_CHANNELS=4    # 4 = xyz + 1 mask channel (use_rgb=False, matches the HL ckpt)

# Video saving for every episode. Lands at <OUTPUT_DIR>/media/.
SAVE_VIDEOS=1
VIDEO_FPS=10

# Where to dump args.json / results.jsonl / summary.json.
# Leave empty ("") to let the Python script auto-generate
#   outputs_eval_gmm/<HL_ckpt_stem>__<LL_dir>_<ckpt>/<timestamp>/
OUTPUT_DIR="GMM_HIGH_LEVEL_2D_DIT_LOW_LEVEL_MUG_CLEANUP_50_SAMPLES_D1_GMM_WCA_State_AdaLN_DINOV2_95_epoch_100_episodes_3RD_SEED"

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

# Same env-var posture as eval_ghost_high_level_2d_dit_low_level.sh.
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

"${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_gmm_high_level_2d_dit_low_level.py \
    --dataset_path         "${DATASET_PATH}"  \
    --high_level_ckpt      "${HL_CKPT}"       \
    --low_level_exp_dir    "${LL_EXP_DIR}"    \
    --low_level_checkpoint "${LL_CKPT}"       \
    --lfd3d_repo           "${LFD3D_REPO}"    \
    --dit_2d_repo          "${DIT_2D_REPO}"   \
    --robosuite_root       "${ROBOSUITE_ROOT}" \
    --hl_in_channels       "${HL_IN_CHANNELS}" \
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
