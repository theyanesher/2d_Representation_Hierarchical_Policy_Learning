#!/usr/bin/env bash
# Hierarchical eval: lfd3d ArticubotNetwork HL + 2D DiT GMM-LL on a MimicGen task.
#
# Kitchen_D1 variant. Sister of eval_gmm_high_level_2d_dit_low_level_kitchen_d1.sh.
# Same env / same action-conversion path. The LL is the GMM-conditioned variant
# trained with:
#   task=MimicGen_Tasks/kitchen_gmm_goal
#   policy.use_goal_cross_attention=true
#   policy.use_weighted_cross_attention=true
#   policy.gmm_top_k=128
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFD3D_REPO="${PROJ_ROOT}/lfd3d/lfd3d"
DIT_2D_REPO="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/diffusion_policy"

# Use the Yufei pixi env's interpreter directly (PyOpenGL 3.1.10 needed for
# mujoco's EGLDeviceEXT; SMITH's own .pixi env has 3.1.0 which is too old).
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/ORIGINAL_DATASET/kitchen_d1.hdf5"

# HL: multimodal Articubot ckpt trained on Kitchen.
HL_CKPT="${PROJ_ROOT}/SMITH_High_Level_FineTune/HIGH_LEVEL_POLICIES/Kitchen_d2/periodic-epoch=epoch=74.ckpt"

# LL: the Kitchen_D1 100-demo run. Hydra exp dir + checkpoint name.
LL_EXP_DIR="${PROJ_ROOT}/SMITH_High_Level_FineTune/LOW_LEVEL_POLICIES/Kitchen_D1/100_demos/DinoV2_GMM_WCA/"
LL_CKPT="epoch_90.ckpt"

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
SEED=100000
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
OUTPUT_DIR="${SCRIPT_DIR}/GMM_HIGH_LEVEL_2D_DIT_LOW_LEVEL_KITCHEN_50_SAMPLES_D1_DINOV2_GMM_WCA_1ST_SEED"

# --------------------------------------------------------------------------- #
# Auto-resume + auto-merge.
#
# Strategy:
#   - Python's eval script opens results.jsonl with mode "w" (truncates) and
#     numbers episodes 1..N internally, so we can't write the resumed batch
#     directly into ${OUTPUT_DIR}. Instead we route it to a sibling temp dir
#     ${OUTPUT_DIR}_RESUME_${COMPLETED}, then fold that back into ${OUTPUT_DIR}
#     after Python finishes (renaming videos by seed so episode_NNN is
#     continuous, rewriting results.jsonl rows, appending).
#   - On startup we also fold in any pre-existing ${OUTPUT_DIR}_RESUME_* dirs
#     (from earlier interrupted runs) so the next COMPLETED count is correct.
#
# Naming: across all merged files the episode index is derived from the seed:
#   ep_index = seed - ORIG_SEED + 1
# --------------------------------------------------------------------------- #

# Snapshot pre-mutation values; do_merge needs ORIG_SEED, the post-Python
# merge needs MAIN_OUTPUT_DIR.
ORIG_SEED=${SEED}
MAIN_OUTPUT_DIR="${OUTPUT_DIR}"

# do_merge SRC_DIR DST_DIR THE_ORIG_SEED
#   Folds SRC_DIR (a resume dir) into DST_DIR (the main dir):
#     - mv+rename SRC/media/*.mp4 and SRC/media_with_goal_overlay/*.mp4 into
#       DST/<same>/ as episode_<seed - THE_ORIG_SEED + 1>_seed_<seed>_<outcome>.mp4
#     - append SRC/results.jsonl rows to DST/results.jsonl with the "episode"
#       field and the "video" / "video_with_goal_overlay" paths rewritten
#     - touch SRC/.merged so future runs skip it
do_merge() {
  local SRC="$1"
  local DST="$2"
  local THE_ORIG_SEED="$3"
  if [[ ! -f "${SRC}/results.jsonl" ]]; then
    echo "[merge] ${SRC} has no results.jsonl, skipping"
    return 0
  fi
  mkdir -p "${DST}/media" "${DST}/media_with_goal_overlay"
  "${ENV_PY}" - "${SRC}" "${DST}" "${THE_ORIG_SEED}" <<'PYEOF'
import json, sys, shutil
from pathlib import Path
src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
orig_seed = int(sys.argv[3])
src_jsonl = src_dir / "results.jsonl"
dst_jsonl = dst_dir / "results.jsonl"
n = 0
with open(src_jsonl) as fi, open(dst_jsonl, "a") as fo:
    for line in fi:
        d = json.loads(line)
        n += 1
        seed = d["seed"]
        new_ep = seed - orig_seed + 1
        outcome = "success" if d["success"] else "failure"
        d["episode"] = new_ep
        for key, sub in (("video", "media"),
                         ("video_with_goal_overlay", "media_with_goal_overlay")):
            old = d.get(key)
            if not old:
                continue
            # Old path is stored relative to the bash cwd (SMITH_MIMICGEN).
            # Try it as-is first, then fall back to (src_dir / sub / basename).
            candidates = [Path(old), src_dir / sub / Path(old).name]
            old_path = next((p for p in candidates if p.exists()), None)
            new_name = f"episode_{new_ep:03d}_seed_{seed}_{outcome}.mp4"
            dst_path = dst_dir / sub / new_name
            if old_path is not None:
                shutil.move(str(old_path), str(dst_path))
            else:
                print(f"[merge][WARN] missing source video for line {n}: {old}")
            d[key] = str(dst_dir / sub / new_name)
        fo.write(json.dumps(d) + "\n")
print(f"[merge] appended {n} rows from {src_jsonl} -> {dst_jsonl}")
PYEOF
  touch "${SRC}/.merged"
  echo "[merge] marked ${SRC}/.merged"
}

# Step 1: fold any leftover ${OUTPUT_DIR}_RESUME_* dirs from earlier runs.
shopt -s nullglob
for resume_dir in "${MAIN_OUTPUT_DIR}_RESUME_"*; do
  if [[ -d "${resume_dir}" && ! -f "${resume_dir}/.merged" ]]; then
    echo "[merge] folding $(basename "${resume_dir}") into ${MAIN_OUTPUT_DIR}"
    do_merge "${resume_dir}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
  fi
done
shopt -u nullglob

# Step 2: count COMPLETED, decide whether to run Python and where it writes.
COMPLETED=0
if [[ -f "${MAIN_OUTPUT_DIR}/results.jsonl" ]]; then
  COMPLETED=$(wc -l < "${MAIN_OUTPUT_DIR}/results.jsonl")
fi

if (( COMPLETED >= N_EPISODES )); then
  echo "[resume] all ${N_EPISODES} episodes already in ${MAIN_OUTPUT_DIR}. Nothing to do."
  exit 0
fi

# PY_OUTPUT_DIR is where the Python script will write. If COMPLETED > 0, this
# is a fresh sibling temp dir; otherwise it's the main dir itself.
PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}"
if (( COMPLETED > 0 )); then
  SEED=$(( SEED + COMPLETED ))
  N_EPISODES=$(( N_EPISODES - COMPLETED ))
  PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}_RESUME_${COMPLETED}"
  echo "[resume] ${COMPLETED} episodes already in ${MAIN_OUTPUT_DIR}."
  echo "[resume] running ${N_EPISODES} more (seeds ${SEED}..$((SEED + N_EPISODES - 1)))."
  echo "[resume] Python writes to ${PY_OUTPUT_DIR}; will fold into ${MAIN_OUTPUT_DIR} on success."
fi

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
export PYTHONPATH="${ROBOSUITE_ROOT}:${PROJ_ROOT}/MimicGen/robosuite-task-zoo:${LFD3D_REPO}/src:${DIT_2D_REPO}:${SMITH_MIMICGEN}:${PYTHONPATH:-}"

OUTPUT_DIR_FLAG=()
if [[ -n "${PY_OUTPUT_DIR}" ]]; then
  OUTPUT_DIR_FLAG=(--output_dir "${PY_OUTPUT_DIR}")
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

# --------------------------------------------------------------------------- #
# Post-run merge: if Python wrote to a sibling resume dir, fold it into the
# main dir (renames videos by seed, appends results.jsonl).
# --------------------------------------------------------------------------- #
if [[ "${PY_OUTPUT_DIR}" != "${MAIN_OUTPUT_DIR}" ]]; then
  echo "[merge] folding ${PY_OUTPUT_DIR} into ${MAIN_OUTPUT_DIR}"
  do_merge "${PY_OUTPUT_DIR}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
fi
