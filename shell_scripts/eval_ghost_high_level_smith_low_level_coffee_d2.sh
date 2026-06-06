#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL (GHOST) + SMITH DP3 LL on a MimicGen task.
#
# Coffee D2 variant. Hybrid of:
#   - eval_ghost_high_level_2d_dit_low_level_coffee_d2_100epoch.sh (GHOST HL + 2D DiT LL)
#   - eval_smith_high_level_low_level_coffee_d2_100_demo_finetune.sh (SMITH HL + SMITH LL)
#
# Heavy training-time deps in the lfd3d package (pytorch_lightning, pytorch3d,
# diffusers, wandb, trimesh, transformers) are stubbed in sys.modules by the
# Python script if missing — they're never reached on the inference path.
# Unlike the 2D-DiT variant, train_ddp is NOT stubbed here — we use the real
# TrainDP3Workspace to load the SMITH LL checkpoint.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths — adjust if any of these dirs move.
# --------------------------------------------------------------------------- #
PROJ_ROOT="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT"

SMITH_MIMICGEN="${PROJ_ROOT}/SMITH_MimicGen/SMITH_on_mimicgen"
SMITH_FINETUNE="${PROJ_ROOT}/SMITH_High_Level_FineTune"
SMITH_ROBOGEN="${SMITH_FINETUNE}/RoboGen-sim2real"
SMITH_FINETUNED="${SMITH_FINETUNE}/SMITH_FINETUNED_MODELS/Coffee_D2"
LFD3D_REPO="${PROJ_ROOT}/lfd3d/lfd3d"

# Use the Yufei pixi env's interpreter directly (PyOpenGL 3.1.10 needed for
# mujoco's EGLDeviceEXT; SMITH's own .pixi env has 3.1.0 which is too old).
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${SMITH_FINETUNE}/ORIGINAL_DATASET/coffee_d2.hdf5"

# GHOST HL ckpt (NO_MULTIMODAL, epoch=74). Matches eval_ghost_high_level_2d_dit_low_level_coffee_d2_100epoch.sh.
HL_CKPT="${SMITH_FINETUNE}/HIGH_LEVEL_POLICIES/Coffee_d2/GHOST_High_Level_NO_MULTIMODAL/periodic-epoch=epoch=19.ckpt"

# SMITH LL training run dir + ckpt. Matches eval_smith_high_level_low_level_coffee_d2_100_demo_finetune.sh.
LL_EXP_DIR="${SMITH_FINETUNED}/LOW_LEVEL_1000_DEMO_TRAINING"      #/LOW_LEVEL"
LL_CKPT="epoch=0300-test_mean_score=-0.000.ckpt" #"epoch=0300-test_mean_score=-0.001.ckpt"

# Optional .npy with a (1152,) text embedding for the HL FiLM block.
# Leave empty to use zeros (matches the NO_MULTIMODAL coffee ckpt's training).
TEXT_EMBED_CACHE=""

# SMITH's external/mimicgen/offcial_robosuite/ is incomplete (no robosuite.models).
# Use the full robosuite checkout that ships under MimicGen/.
ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# SigLIP category index for SMITH LL conditioning. 0 = "open the storage furniture"
# (the default for Coffee_D2 since its path matches no category substring in
# dataset_from_disk.py).
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
HL_IN_CHANNELS=4    # 4 = xyz + 1 mask channel (use_rgb=False, matches the coffee ckpt)
HL_ARGMAX_WEIGHT=1  # 1 = argmax over softmaxed scene weights; 0 = multinomial

# Video saving for every episode. Lands at <OUTPUT_DIR>/media/.
SAVE_VIDEOS=1
VIDEO_FPS=10

# Where to dump args.json / results.jsonl / summary.json.
OUTPUT_DIR="${SMITH_MIMICGEN}/GHOST_HL_SMITH_LL_COFFEE_D2_50_SAMPLES_1000_DEMO_FINETUNING_300"

# --------------------------------------------------------------------------- #
# Sanity checks before launching
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" "${HL_CKPT}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${LFD3D_REPO}/src/lfd3d/models/articubot.py" \
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

if [[ -n "${TEXT_EMBED_CACHE}" && ! -e "${TEXT_EMBED_CACHE}" ]]; then
  echo "[ERROR] TEXT_EMBED_CACHE set but missing: ${TEXT_EMBED_CACHE}" >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
cd "${SMITH_MIMICGEN}"

# Same env-var posture as the other eval scripts.
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}

# Required by diffusion_policy_3d.policy.dp3.DP3.__init__ — loads SigLIP from
# ${PROJECT_DIR}/siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt.
export PROJECT_DIR="${SMITH_FINETUNE}"

# Cosmetic PYTHONPATH export — the Python script prepends these itself via
# _bootstrap_paths, but non-pixi runs benefit from having them on the env too.
# Order mirrors _bootstrap_paths (later wins on sys.path insert(0)):
SMITH_DP3="${SMITH_ROBOGEN}/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy"
export PYTHONPATH="${SMITH_ROBOGEN}:${SMITH_DP3}:${SMITH_MIMICGEN}:${LFD3D_REPO}/src:${PYTHONPATH:-}"

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

"${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_ghost_high_level_smith_low_level.py \
    --dataset_path         "${DATASET_PATH}"   \
    --high_level_ckpt      "${HL_CKPT}"        \
    --low_level_exp_dir    "${LL_EXP_DIR}"     \
    --low_level_checkpoint "${LL_CKPT}"        \
    --lfd3d_repo           "${LFD3D_REPO}"     \
    --smith_robogen_repo   "${SMITH_ROBOGEN}"  \
    --robosuite_root       "${ROBOSUITE_ROOT}" \
    --hl_in_channels       "${HL_IN_CHANNELS}" \
    --hl_argmax_weight     "${HL_ARGMAX_WEIGHT}" \
    --cat_idx              "${CAT_IDX}"        \
    --n_episodes           "${N_EPISODES}"     \
    --max_steps            "${MAX_STEPS}"      \
    --seed                 "${SEED}"           \
    --n_obs_steps          "${N_OBS_STEPS}"    \
    --n_action_steps       "${N_ACTION_STEPS}" \
    --camera_h             "${CAMERA_H}"       \
    --camera_w             "${CAMERA_W}"       \
    "${TEXT_EMBED_FLAG[@]}"                    \
    "${VIDEO_FLAG[@]}"                         \
    "${OUTPUT_DIR_FLAG[@]}"
