#!/usr/bin/env bash
# Hierarchical eval: SMITH HL + 2D DiT LL on a MimicGen task.
#
# - Adds the two foreign code roots to PYTHONPATH so the LL ckpt unpickles
#   and the HL model class resolves.
# - Runs from SMITH_MimicGen/SMITH_on_mimicgen so eval_smith_utils.py is importable.
# - Uses the SMITH_MimicGen pixi env. If the LL ckpt fails to unpickle because of
#   a missing dependency (diffusers, transformers, etc.), either install it into
#   this env or switch to the 2D codebase's pixi env (which has those deps but
#   may need robomimic/robosuite/mimicgen added).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths — adjust if any of these dirs move.
# --------------------------------------------------------------------------- #
PROJ_ROOT="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT"

SMITH_MIMICGEN="${PROJ_ROOT}/SMITH_MimicGen/SMITH_on_mimicgen"
SMITH_HL_REPO="${PROJ_ROOT}/SMITH_High_Level_FineTune/RoboGen-sim2real"
DIT_2D_REPO="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/diffusion_policy"

# Use the Yufei pixi env's interpreter directly (skips `pixi run` overhead and
# guarantees we land in the env that has PyOpenGL 3.1.10). SMITH's own .pixi env
# has PyOpenGL 3.1.0 which is too old for mujoco's EGLDeviceEXT lookup.
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/ORIGINAL_DATASET/coffee_d2.hdf5"
HL_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/exps/2026-04-27finetune_coffee_task/model_97501.pth"
LL_EXP_DIR="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.04.29/08.39.57_coffee_goal_gripper_DIT_coffee_goal_gripper"
LL_CKPT="epoch_60.ckpt"
SIGLIP_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt"

# SMITH's external/mimicgen/offcial_robosuite/ is incomplete (no robosuite.models).
# Use the full robosuite checkout that ships under MimicGen/. Parent dir of robosuite/.
ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# --------------------------------------------------------------------------- #
# Eval knobs
# --------------------------------------------------------------------------- #
CAT_IDX=0           # Coffee_Task fine-tune set this in dataset_from_disk.py:135
N_EPISODES=50
MAX_STEPS=400
SEED=100000
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256

# Video saving for successful episodes. Lands at <OUTPUT_DIR>/media/.
# Set SAVE_VIDEOS=0 to disable. VIDEO_FPS matches SMITH's articubot_pcd_runner default.
SAVE_VIDEOS=1
VIDEO_FPS=10

# Where to dump args.json / results.jsonl / summary.json.
# Leave empty ("") to let the Python script auto-generate
#   outputs_eval/<HL_dir>__<LL_dir>_<ckpt>/<timestamp>/
OUTPUT_DIR="SMITH_COFEE_FINATUNED_HIGH_LEVEL_GROOT_3D_LOW_LEVEL_50_SAMPLES_D2"

# --------------------------------------------------------------------------- #
# Sanity checks before launching
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" "${HL_PATH}" "${SIGLIP_PATH}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${SMITH_HL_REPO}/test_PointNet2/model_invariant.py" \
         "${DIT_2D_REPO}/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py" \
         "${ROBOSUITE_ROOT}/robosuite/models/__init__.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
cd "${SMITH_MIMICGEN}"

# These mirror env vars used elsewhere in this codebase (e.g. fine_tune_command.txt).
# PYTHONNOUSERSITE=1 is critical: ~/.local/lib/python3.10/site-packages/OpenGL
# (an older PyOpenGL) shadows the env's 3.1.10 if user-site is enabled.
# PYOPENGL_PLATFORM=egl forces PyOpenGL to load EGL platform stubs at import time
# (otherwise it picks GLX and EGLDeviceEXT can't be resolved).
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}

# NOTE: setting PYTHONPATH here is mostly cosmetic — SMITH's pixi.toml has an
# [activation] env entry that overwrites it. The Python script prepends the
# foreign roots (HL repo, 2D LL repo) and SMITH's vendored external/ deps
# (robomimic, mimicgen, offcial_robosuite) to sys.path itself, which survives
# pixi activation. We still export it so non-pixi runs work.
export PYTHONPATH="${SMITH_HL_REPO}:${DIT_2D_REPO}:${SMITH_MIMICGEN}:${PYTHONPATH:-}"

# Optional --output_dir flag (only added if OUTPUT_DIR is non-empty).
OUTPUT_DIR_FLAG=()
if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_FLAG=(--output_dir "${OUTPUT_DIR}")
fi

# Video flag. The Python script uses argparse.BooleanOptionalAction, so the negative
# form is --no-save-videos (with a hyphen).
if [[ "${SAVE_VIDEOS}" == "0" ]]; then
  VIDEO_FLAG=(--no-save-videos)
else
  VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
fi

# Use the Yufei pixi env's python directly (see ENV_PY definition above).
"${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_2d_dit_mimicgen.py \
    --dataset_path        "${DATASET_PATH}"  \
    --high_level_path     "${HL_PATH}"       \
    --low_level_exp_dir   "${LL_EXP_DIR}"    \
    --low_level_checkpoint "${LL_CKPT}"      \
    --siglip_features_path "${SIGLIP_PATH}"  \
    --smith_hl_repo       "${SMITH_HL_REPO}" \
    --dit_2d_repo         "${DIT_2D_REPO}"   \
    --robosuite_root      "${ROBOSUITE_ROOT}" \
    --cat_idx             "${CAT_IDX}"       \
    --n_episodes          "${N_EPISODES}"    \
    --max_steps           "${MAX_STEPS}"     \
    --seed                "${SEED}"          \
    --n_obs_steps         "${N_OBS_STEPS}"   \
    --n_action_steps      "${N_ACTION_STEPS}" \
    --camera_h            "${CAMERA_H}"      \
    --camera_w            "${CAMERA_W}"      \
    "${VIDEO_FLAG[@]}"                       \
    "${OUTPUT_DIR_FLAG[@]}"
