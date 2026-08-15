#!/usr/bin/env bash
# APPROACH 2 eval on Hammer_Cleanup_D1: 2D DiT low-level policy ALONE.
#
# There is NO high-level policy here. The Approach 2 LL is never given a goal —
# goal_gripper_pts only supervised its visual representation during training via
# the auxiliary GMM head, so at rollout the policy is self-contained.
#
# Single script, runs seeds 100000 / 150000 / 250000 SEQUENTIALLY.
# Per-seed outputs land in:
#   ${SCRIPT_DIR}/APPROACH2_2D_DIT_LOW_LEVEL_HAMMER_CLEANUP_50_SAMPLES_D1_DINOV2_c1_0.1_<NTH>_SEED/
# Each seed has its own auto-resume bookkeeping (do_merge below); an interrupted
# run can be re-launched and continues from where it left off.
#
# Layout written per seed (identical to the hierarchical evals, so downstream
# analysis and the merge logic are unchanged):
#   args.json  results.jsonl  summary.json  media/*.mp4  media_with_goal_overlay/
# media_with_goal_overlay/ stays empty — Approach 2 has no goal to overlay.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths (all on PSC)
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INFERENCE_ROOT="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Mimicgen_Inference/2d_Representation_Hierarchical_Policy_Learning"
LL_REPO="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"
DIT_2D_REPO="${LL_REPO}/diffusion_policy"

ENV_PY="${INFERENCE_ROOT}/.pixi/envs/default/bin/python"

DATASET_PATH="${INFERENCE_ROOT}/DATASET/Hammer_D1_Dataset/core/hammer_cleanup_d1.hdf5"

# LL: 100-demo Approach 2 run on Hammer_Cleanup_D1, c1 = 0.1.
LL_EXP_DIR="${LL_REPO}/outputs/2026.08.12/21.38.51_hammercleanup_D1_APPROACH2_c1_0.1_100demo_hammercleanup_D1_goal_gmm_aux"
LL_CKPT="epoch_99.ckpt"

# robosuite / mimicgen / robosuite-task-zoo are installed into the inference
# pixi env, so nothing extra is needed on PYTHONPATH for them.
ROBOSUITE_ROOT=""

# --------------------------------------------------------------------------- #
# Eval knobs  (kept identical to the Approach 1 evals so numbers are comparable)
# --------------------------------------------------------------------------- #
N_EPISODES=50
MAX_STEPS=800
N_OBS_STEPS=2
N_ACTION_STEPS=8
CAMERA_H=256
CAMERA_W=256

SAVE_VIDEOS=1
VIDEO_FPS=10

OUTPUT_BASE="APPROACH2_2D_DIT_LOW_LEVEL_HAMMER_CLEANUP_50_SAMPLES_D1_DINOV2_c1_0.1"

# --------------------------------------------------------------------------- #
# do_merge SRC DST ORIG_SEED  — folds a sibling _RESUME_<N> dir into the main per-seed dir.
# --------------------------------------------------------------------------- #
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
# Sanity checks
# --------------------------------------------------------------------------- #
for f in "${DATASET_PATH}" \
         "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${DIT_2D_REPO}/diffusion_policy/policy/flow_matching_dit_goal_gmm_policy.py" \
         "${INFERENCE_ROOT}/eval_smith_utils.py" \
         "${INFERENCE_ROOT}/third_party/robogen/robogen_utils.py" \
         "${LL_REPO}/manipulation/utils.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done

# --------------------------------------------------------------------------- #
# Launch env
# --------------------------------------------------------------------------- #
cd "${INFERENCE_ROOT}"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}
# $HOME is quota-tight; keep scratch off it so mujoco/ffmpeg temp writes can't fail.
export TMPDIR="${TMPDIR:-/tmp/approach2_eval_$$}"
mkdir -p "${TMPDIR}"
# INFERENCE_ROOT: eval_smith_utils + third_party.robogen
# LL_REPO       : manipulation.utils (imported by robogen_utils)
export PYTHONPATH="${INFERENCE_ROOT}:${LL_REPO}:${PYTHONPATH:-}"

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
    echo "[resume] ${COMPLETED} episodes already in ${MAIN_OUTPUT_DIR}."
    echo "[resume] running ${CUR_N_EP} more (seeds ${CUR_SEED}..$(( CUR_SEED + CUR_N_EP - 1 )))."
    echo "[resume] Python writes to ${PY_OUTPUT_DIR}; will fold into ${MAIN_OUTPUT_DIR} on success."
  fi

  local VIDEO_FLAG
  if [[ "${SAVE_VIDEOS}" == "0" ]]; then
    VIDEO_FLAG=(--no-save-videos)
  else
    VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
  fi
  local ROBOSUITE_FLAG=()
  if [[ -n "${ROBOSUITE_ROOT}" ]]; then
    ROBOSUITE_FLAG=(--robosuite_root "${ROBOSUITE_ROOT}")
  fi

  "${ENV_PY}" external/mimicgen/mimicgen/scripts/eval_approach2_2d_dit_low_level.py \
      --dataset_path         "${DATASET_PATH}"   \
      --low_level_exp_dir    "${LL_EXP_DIR}"     \
      --low_level_checkpoint "${LL_CKPT}"        \
      --dit_2d_repo          "${DIT_2D_REPO}"    \
      "${ROBOSUITE_FLAG[@]}"                     \
      --n_episodes           "${CUR_N_EP}"       \
      --max_steps            "${MAX_STEPS}"      \
      --seed                 "${CUR_SEED}"       \
      --n_obs_steps          "${N_OBS_STEPS}"    \
      --n_action_steps       "${N_ACTION_STEPS}" \
      --camera_h             "${CAMERA_H}"       \
      --camera_w             "${CAMERA_W}"       \
      "${VIDEO_FLAG[@]}"                         \
      --output_dir           "${PY_OUTPUT_DIR}"

  if [[ "${PY_OUTPUT_DIR}" != "${MAIN_OUTPUT_DIR}" ]]; then
    echo "[merge] folding ${PY_OUTPUT_DIR} into ${MAIN_OUTPUT_DIR}"
    do_merge "${PY_OUTPUT_DIR}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
  fi
}

# --------------------------------------------------------------------------- #
# Run all three seeds sequentially
# --------------------------------------------------------------------------- #
run_one_seed 1ST 100000
run_one_seed 2ND 150000
run_one_seed 3RD 250000

echo
echo "All seeds done. Outputs:"
for s in 1ST 2ND 3RD; do
  echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_${s}_SEED"
done
