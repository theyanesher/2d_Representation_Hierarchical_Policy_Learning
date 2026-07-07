#!/usr/bin/env bash
# Hierarchical eval on Coffee_D2 with RDP-trained HL + goal_gripper (ghost) LL.
# Runs seeds 100000 / 150000 / 250000 SEQUENTIALLY, each with independent auto-resume.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJ_ROOT="/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT"

SMITH_MIMICGEN="${PROJ_ROOT}/SMITH_MimicGen/SMITH_on_mimicgen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFD3D_REPO="${PROJ_ROOT}/lfd3d/lfd3d"
DIT_2D_REPO="${PROJ_ROOT}/2D_Hierarchical_Policy_Learning_Github/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/diffusion_policy"

# Yufei pixi env interpreter (PyOpenGL 3.1.10 for mujoco EGLDeviceEXT).
ENV_PY="${PROJ_ROOT}/Yufei_Data_Generation_Code_FINAL/cleaned_smith_real_world_inference/.pixi/envs/default/bin/python"

DATASET_PATH="${PROJ_ROOT}/SMITH_High_Level_FineTune/ORIGINAL_DATASET/coffee_d2.hdf5"

# HL: RDP-trained multimodal Articubot ckpt for Coffee_d2.
HL_CKPT="${PROJ_ROOT}/SMITH_High_Level_FineTune/HIGH_LEVEL_POLICIES/Coffee_d2_RDP/periodic-epoch=epoch=44.ckpt"

# LL: 100-demo goal_gripper (ghost) run (RDP LL tree).
LL_EXP_DIR="${PROJ_ROOT}/SMITH_High_Level_FineTune/LOW_LEVEL_POLICIES/Coffee_d2_RDP/100_Demos/DinoV2_goal_gripper/"
LL_CKPT="epoch_100.ckpt"

# Optional .npy with a (1152,) text embedding for the HL FiLM block.
TEXT_EMBED_CACHE=""

ROBOSUITE_ROOT="${PROJ_ROOT}/MimicGen/robosuite"

# --------------------------------------------------------------------------- #
# Eval knobs
# --------------------------------------------------------------------------- #
N_EPISODES=50
MAX_STEPS=400
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256
HL_IN_CHANNELS=4    # 4 = xyz + 1 mask channel (use_rgb=False)
HL_ARGMAX_WEIGHT=1  # 1 = argmax over softmaxed scene weights; 0 = multinomial

SAVE_VIDEOS=1
VIDEO_FPS=10

OUTPUT_BASE="GHOST_HIGH_LEVEL_2D_DIT_LOW_LEVEL_COFFEE_RDP_50_SAMPLES_D2_DINOV2_goal_gripper"

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
# Launch env (Python entrypoint lives under SMITH_MIMICGEN)
# --------------------------------------------------------------------------- #
cd "${SMITH_MIMICGEN}"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}
# Coffee envs live in base robosuite (no robosuite_task_zoo needed).
export PYTHONPATH="${LFD3D_REPO}/src:${DIT_2D_REPO}:${SMITH_MIMICGEN}:${PYTHONPATH:-}"

run_one_seed() {
  local sfx="$1" seed="$2"
  local ORIG_SEED=${seed}
  local MAIN_OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${sfx}_SEED"

  echo
  echo "==========================================================================="
  echo "[seed ${sfx}] SEED=${seed}  OUTPUT=${MAIN_OUTPUT_DIR}"
  echo "==========================================================================="

  shopt -s nullglob
  for resume_dir in "${MAIN_OUTPUT_DIR}_RESUME_"*; do
    if [[ -d "${resume_dir}" && ! -f "${resume_dir}/.merged" ]]; then
      echo "[merge] folding $(basename "${resume_dir}") into ${MAIN_OUTPUT_DIR}"
      do_merge "${resume_dir}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
    fi
  done
  shopt -u nullglob

  local COMPLETED=0
  if [[ -f "${MAIN_OUTPUT_DIR}/results.jsonl" ]]; then
    COMPLETED=$(wc -l < "${MAIN_OUTPUT_DIR}/results.jsonl")
  fi

  if (( COMPLETED >= N_EPISODES )); then
    echo "[resume] all ${N_EPISODES} episodes already in ${MAIN_OUTPUT_DIR}. Skipping."
    return 0
  fi

  local PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}"
  local CUR_SEED=${seed}
  local CUR_N_EP=${N_EPISODES}
  if (( COMPLETED > 0 )); then
    CUR_SEED=$(( seed + COMPLETED ))
    CUR_N_EP=$(( N_EPISODES - COMPLETED ))
    PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}_RESUME_${COMPLETED}"
    echo "[resume] ${COMPLETED} already done; running ${CUR_N_EP} more (seeds ${CUR_SEED}..$(( CUR_SEED + CUR_N_EP - 1 )))."
  fi

  local OUTPUT_DIR_FLAG=(--output_dir "${PY_OUTPUT_DIR}")
  local VIDEO_FLAG
  if [[ "${SAVE_VIDEOS}" == "0" ]]; then
    VIDEO_FLAG=(--no-save-videos)
  else
    VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
  fi
  local TEXT_EMBED_FLAG=()
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
        --n_episodes           "${CUR_N_EP}"     \
        --max_steps            "${MAX_STEPS}"     \
        --seed                 "${CUR_SEED}"      \
        --n_obs_steps          "${N_OBS_STEPS}"   \
        --n_action_steps       "${N_ACTION_STEPS}" \
        --camera_h             "${CAMERA_H}"      \
        --camera_w             "${CAMERA_W}"      \
        "${TEXT_EMBED_FLAG[@]}"                   \
        "${VIDEO_FLAG[@]}"                        \
        "${OUTPUT_DIR_FLAG[@]}"

  if [[ "${PY_OUTPUT_DIR}" != "${MAIN_OUTPUT_DIR}" ]]; then
    echo "[merge] folding ${PY_OUTPUT_DIR} into ${MAIN_OUTPUT_DIR}"
    do_merge "${PY_OUTPUT_DIR}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
  fi
}

# --------------------------------------------------------------------------- #
# Run all three seeds sequentially
# --------------------------------------------------------------------------- #
HL_ARGMAX_WEIGHT=1  # only consumed by eval_ghost_*.py
run_one_seed 1ST 100000
run_one_seed 2ND 150000
run_one_seed 3RD 250000

echo
echo "All seeds done. Outputs:"
for s in 1ST 2ND 3RD; do
  echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_${s}_SEED"
done
