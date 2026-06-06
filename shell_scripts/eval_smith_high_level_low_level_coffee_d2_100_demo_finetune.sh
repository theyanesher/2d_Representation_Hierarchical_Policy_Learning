#!/usr/bin/env bash
# Hierarchical eval: SMITH HL (PointNet2_super_multitask, SigLIP-conditioned)
# + SMITH LL (diffusion_policy_3d.DP3, Act3D encoder) on a MimicGen task.
#
# Coffee D2 variant. Mirrors eval_ghost_high_level_2d_dit_low_level_coffee_d2_*.sh
# but loads SMITH-class models from SMITH_PRETRAINED_MODELS/ instead of lfd3d HL
# + 2D DiT LL.
#
# For now the HL/LL paths below point at SMITH_PRETRAINED_MODELS/. When you
# swap to the finetuned checkpoints (same architecture), only HL_CKPT and
# LL_EXP_DIR need to change.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths — adjust if any of these dirs move.
# --------------------------------------------------------------------------- #
PROJ_ROOT="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT"

SMITH_MIMICGEN="${PROJ_ROOT}/SMITH_MimicGen/SMITH_on_mimicgen"
SMITH_FINETUNE="${PROJ_ROOT}/SMITH_High_Level_FineTune"
SMITH_ROBOGEN="${SMITH_FINETUNE}/RoboGen-sim2real"
SMITH_FINETUNED="${SMITH_FINETUNE}/SMITH_FINETUNED_MODELS/Coffee_D2"

# Use the Yufei pixi env's interpreter directly (PyOpenGL 3.1.10 needed for
# mujoco's EGLDeviceEXT; SMITH's own .pixi env has 3.1.0 which is too old).
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${SMITH_FINETUNE}/ORIGINAL_DATASET/coffee_d2.hdf5"

# SMITH HL .pth (config.json must sit next to it — train_multitask_ddp...py writes it).
HL_CKPT="${SMITH_FINETUNED}/HIGH_LEVEL/model_67501.pth"

# SMITH LL training run dir: must contain .hydra/config.yaml and checkpoints/<LL_CKPT>.
LL_EXP_DIR="${SMITH_FINETUNED}/LOW_LEVEL"
LL_CKPT="epoch=0300-test_mean_score=-0.001.ckpt"

# SMITH's external/mimicgen/offcial_robosuite/ is incomplete (no robosuite.models).
# Use the full robosuite checkout that ships under MimicGen/.
ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# SigLIP category index for HL conditioning. 0 = "open the storage furniture"
# (the default for Coffee_D2 since its path matches no category substring in
# dataset_from_disk.py — see the analysis in this branch's commit history).
CAT_IDX=0

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

SAVE_VIDEOS=1
VIDEO_FPS=10

OUTPUT_DIR="${SMITH_MIMICGEN}/SMITH_HIGH_LEVEL_SMITH_LOW_LEVEL_COFFEE_D2_50_SAMPLES_100_DEMO_FINETUNE"

# --------------------------------------------------------------------------- #
# Sanity checks before launching
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" "${HL_CKPT}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${SMITH_ROBOGEN}/test_PointNet2/model_invariant.py" \
         "${SMITH_ROBOGEN}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/train_ddp.py" \
         "${SMITH_FINETUNE}/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt" \
         "${ROBOSUITE_ROOT}/robosuite/models/__init__.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

# Config.json must sit next to the HL .pth — the SMITH HL loader reads it to
# rebuild the model with the right siglip/layernorm/first_sa_point options.
HL_CONFIG="$(dirname "${HL_CKPT}")/config.json"
if [[ ! -e "${HL_CONFIG}" ]]; then
  echo "[ERROR] missing HL config.json next to checkpoint: ${HL_CONFIG}" >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
cd "${SMITH_MIMICGEN}"

export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}

# Required by diffusion_policy_3d.policy.dp3.DP3.__init__ — loads SigLIP from
# ${PROJECT_DIR}/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt.
export PROJECT_DIR="${SMITH_FINETUNE}"

# PYTHONPATH: SMITH training-side dp3 and RoboGen-sim2real must come FIRST so
# that `from test_PointNet2.model_invariant import PointNet2_super_multitask`
# resolves to the multitask version (not the cut-down third_party/robogen one
# in SMITH_on_mimicgen). The Python script also prepends these via sys.path,
# but we export here for non-pixi runs.
SMITH_DP3="${SMITH_ROBOGEN}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy"
export PYTHONPATH="${SMITH_DP3}:${SMITH_ROBOGEN}:${SMITH_MIMICGEN}:${PYTHONPATH:-}"

OUTPUT_DIR_FLAG=()
if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_FLAG=(--output_dir "${OUTPUT_DIR}")
fi

if [[ "${SAVE_VIDEOS}" == "0" ]]; then
  VIDEO_FLAG=(--no-save-videos)
else
  VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
fi

"${ENV_PY}" eval_smith.py \
    --dataset_path         "${DATASET_PATH}"   \
    --high_level_ckpt      "${HL_CKPT}"        \
    --low_level_exp_dir    "${LL_EXP_DIR}"     \
    --low_level_checkpoint "${LL_CKPT}"        \
    --smith_robogen_repo   "${SMITH_ROBOGEN}"  \
    --robosuite_root       "${ROBOSUITE_ROOT}" \
    --cat_idx              "${CAT_IDX}"        \
    --n_episodes           "${N_EPISODES}"     \
    --max_steps            "${MAX_STEPS}"      \
    --seed                 "${SEED}"           \
    --n_obs_steps          "${N_OBS_STEPS}"    \
    --n_action_steps       "${N_ACTION_STEPS}" \
    --camera_h             "${CAMERA_H}"       \
    --camera_w             "${CAMERA_W}"       \
    "${VIDEO_FLAG[@]}"                         \
    "${OUTPUT_DIR_FLAG[@]}"
