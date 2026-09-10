#!/usr/bin/env bash
# Generic APPROACH 1 (Articubot HL + 2D DiT LL) eval launcher, PARALLEL.
#
# This script is self-locating: drop (or symlink) it inside ANY Approach 1
# policy run directory under approach1_baselines_best_ckpt/<Task_D1>/ and it
# derives everything it needs from that directory:
#   - run name / task / variant                   <- .hydra/config.yaml
#   - LL family (which evaluator to use)          <- .hydra/config.yaml:
#       goal_gripper  (single subgoal, dinov2)    -> eval_ghost_high_level_parallel_2d_dit_low_level.py
#       rope          (GMM+WCA, rope4d_grounded*) -> eval_gmm_high_level_rope_2d_dit_low_level.py
#   - n_obs_steps, n_action_steps                 <- .hydra/config.yaml
#   - dataset hdf5                                <- task -> DATASET_ROOT
#   - low-level checkpoints                       <- ./checkpoints/*.ckpt
#   - high-level (Articubot) checkpoint           <- ../High_Level_Policy/*.ckpt
#                                                    (newest epoch; --hl-ckpt overrides)
#
# Runs LOCALLY (eval pixi env + theya_* worktrees, see CONFIG). Each
# (checkpoint, seed) pair runs the family's parallel evaluator with --num_envs
# workers and records only the first --num-video-episodes.
#
# Usage:
#   ./eval.sh                          # newest checkpoint, all default seeds
#   ./eval.sh --list                   # show what was derived, run nothing
#   ./eval.sh --all                    # every checkpoint in checkpoints/
#   ./eval.sh -c epoch_60.ckpt -c 99   # specific checkpoints ("99" == epoch_99)
#   ./eval.sh --last 2                 # the 2 newest checkpoints (by epoch)
#   ./eval.sh --seeds 100000,150000    # subset of seeds
#   ./eval.sh --episodes 20 --no-videos
#   ./eval.sh --num-envs 8
#   ./eval.sh --mode sequential        # goal_gripper only: old single-env ghost eval (A/B reference)
#   ./eval.sh --dry-run                # print the commands only
#   ./eval.sh --summary-only           # rebuild summaries from existing results
#
# Output layout (per checkpoint), next to the checkpoints:
#   <run>/evaluations/<ckpt>/<eval_name>/<eval_name>_<CKPT>_<seed>_SEED/{results.jsonl,summary.json,media/}
#   <run>/evaluations/<ckpt>/<eval_name>/{eval.log,summary_all_seeds.json,summary_all_seeds.txt}
#   <run>/evaluations/<ckpt>/summary_all_seeds.{json,txt}
#
# Resumable: a (ckpt, seed) pair with N_EPISODES rows in results.jsonl is
# skipped; a partial one is continued from the next seed into a sibling
# _RESUME_<n> dir and folded back (do_merge below).
#
# Every derived value can still be overridden by flag or environment variable
# (see the CONFIG block and parse_args below).

set -euo pipefail

# =============================================================================
# CONFIG -- defaults only; anything here can be overridden by flags/env.
# =============================================================================
INFERENCE_ROOT="${INFERENCE_ROOT:-/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning}"
LFD3D_REPO="${LFD3D_REPO:-/home/theyanesh/worktrees/theya_high_level_lfd3d}"
DIT_2D_REPO="${DIT_2D_REPO:-/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference/diffusion_policy}"
ROBOSUITE_ROOT="${ROBOSUITE_ROOT:-}"
DATASET_ROOT="${DATASET_ROOT:-/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core}"

MODE="${MODE:-parallel}"            # parallel   -> the LL family's parallel evaluator (see FAMILY below)
                                    # sequential -> eval_ghost_high_level_2d_dit_low_level.py (goal_gripper only)
SEEDS_DEFAULT=(100000 150000 250000)
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-800}"
NUM_ENVS="${NUM_ENVS:-8}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-fp32}"
CAMERA_H="${CAMERA_H:-256}"
CAMERA_W="${CAMERA_W:-256}"
HL_IN_CHANNELS="${HL_IN_CHANNELS:-4}"    # 4 = xyz + 1 mask channel (use_rgb=False)
HL_ARGMAX_WEIGHT="${HL_ARGMAX_WEIGHT:-1}" # 1 = argmax over scene anchors; 0 = multinomial
TEXT_EMBED_CACHE="${TEXT_EMBED_CACHE:-}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-4}"
VIDEO_FPS="${VIDEO_FPS:-10}"
# n_obs_steps / n_action_steps default to the training config; override with flags.
N_OBS_STEPS="${N_OBS_STEPS:-}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
DATASET_PATH="${DATASET_PATH:-}"
HL_CKPT="${HL_CKPT:-}"
# =============================================================================

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# The policy run dir is the dir holding this script (or symlink), unless it
# lives in the archived evaluations/<ckpt>/<eval_name>/ copy -- then walk up.
LL_EXP_DIR="$(cd "$(dirname "${SELF}")" && pwd)"
while [[ ! -d "${LL_EXP_DIR}/checkpoints" || ! -f "${LL_EXP_DIR}/.hydra/config.yaml" ]]; do
  parent="$(dirname "${LL_EXP_DIR}")"
  [[ "${parent}" != "${LL_EXP_DIR}" ]] || {
    echo "[ERROR] could not locate a policy run dir (needs checkpoints/ and .hydra/config.yaml) above ${SELF}" >&2
    exit 1
  }
  LL_EXP_DIR="${parent}"
done
LL_RUN_NAME="$(basename "${LL_EXP_DIR}")"
HYDRA_CFG="${LL_EXP_DIR}/.hydra/config.yaml"
CKPT_DIR="${LL_EXP_DIR}/checkpoints"
TASK_DIR="$(dirname "${LL_EXP_DIR}")"          # e.g. .../Kitchen_D1
HL_CKPT_DIR="${HL_CKPT_DIR:-${TASK_DIR}/High_Level_Policy}"

yaml_top_level() {  # yaml_top_level <key> -- first top-level "key: value"
  sed -n "s/^$1:[[:space:]]*//p" "${HYDRA_CFG}" | head -1 | tr -d '\r'
}

# ---- derive run identity ----------------------------------------------------
# The run dirs written by all_download.sh are named <task>_d1_<variant>
# (kitchen_d1_awe_grip_goal_gripper, hammer_cleanup_d1_awe_grip_wca_rope_goals, ...).
# Prefer that; fall back to the training config's `name:` for foreign dirs.
RUN_CFG_NAME="$(yaml_top_level name)"          # e.g. kitchen_D1_goal_gripper_awe_grip_100demo_dinov2_DIT
[[ -n "${RUN_CFG_NAME}" ]] || RUN_CFG_NAME="${LL_RUN_NAME}"

if [[ "${LL_RUN_NAME}" == *_d1_* ]]; then
  TASK_KEY="${LL_RUN_NAME%%_d1_*}_d1"                              # kitchen_d1
  VARIANT_LABEL="${LL_RUN_NAME#*_d1_}"                              # awe_grip_goal_gripper
elif [[ "${RUN_CFG_NAME}" == *_goal_gripper* ]]; then
  TASK_KEY="${RUN_CFG_NAME%%_goal_gripper*}"                        # kitchen_D1
  VARIANT_LABEL="${RUN_CFG_NAME#*_goal_gripper}"; VARIANT_LABEL="${VARIANT_LABEL#_}"
  VARIANT_LABEL="${VARIANT_LABEL%%_100demo*}"; VARIANT_LABEL="${VARIANT_LABEL%%_dinov2*}"
  VARIANT_LABEL="${VARIANT_LABEL:+${VARIANT_LABEL}_}goal_gripper"
else
  TASK_KEY="${LL_RUN_NAME}"
  VARIANT_LABEL="unknown"
fi
TASK_LABEL="$(printf '%s' "${TASK_KEY}" | tr '[:lower:]' '[:upper:]')"

# ---- LL family -> evaluator -------------------------------------------------
#   goal_gripper : dinov2 encoder, single goal_gripper_pts subgoal, no WCA
#                  -> GHOST evaluator (argmax Articubot subgoal)
#   rope         : rope4d_grounded[_goals] encoder + GMM/WCA
#                  -> RoPE GMM evaluator (full GMM injected + depth/intrinsics)
#   gmm          : dinov2 encoder + GMM/WCA (no rope)
#                  -> standard GMM parallel evaluator
# FAMILY=<name> in the environment overrides the detection.
yaml_policy_field() {  # first "  <key>: value" (2-space indent, under policy:)
  sed -n "s/^  $1:[[:space:]]*//p" "${HYDRA_CFG}" | head -1 | tr -d '\r'
}
LL_WCA="$(yaml_policy_field use_weighted_cross_attention)"
LL_TOPK="$(yaml_policy_field gmm_top_k)"
LL_ENC="$(yaml_policy_field visual_encoder_type)"
LL_HAS_GMM=0
if [[ "${LL_WCA}" == "true" || ( -n "${LL_TOPK}" && "${LL_TOPK}" != "null" ) ]]; then LL_HAS_GMM=1; fi
if [[ -z "${FAMILY:-}" ]]; then
  if [[ "${LL_ENC}" == *rope* ]]; then
    FAMILY="rope"
  elif (( LL_HAS_GMM )); then
    FAMILY="gmm"
  elif [[ "${RUN_CFG_NAME}" == *goal_gripper* || "${LL_RUN_NAME}" == *goal_gripper* ]]; then
    FAMILY="goal_gripper"
  else
    echo "[ERROR] cannot tell the LL family of ${LL_RUN_NAME}" >&2
    echo "        name=${RUN_CFG_NAME} use_weighted_cross_attention=${LL_WCA:-?} gmm_top_k=${LL_TOPK:-?} visual_encoder_type=${LL_ENC:-?}" >&2
    echo "        Set FAMILY=goal_gripper|rope|gmm to override." >&2
    exit 1
  fi
fi
if [[ "${FAMILY}" == "rope" && "${LL_ENC}" != *rope* ]]; then
  echo "[ERROR] FAMILY=rope but visual_encoder_type=${LL_ENC:-?} is not a rope4d encoder" >&2; exit 1
fi
if [[ "${FAMILY}" == "goal_gripper" && ( LL_HAS_GMM -eq 1 || "${LL_ENC}" == *rope* ) ]]; then
  echo "[ERROR] FAMILY=goal_gripper but this LL uses GMM/WCA (${LL_WCA:-?}, top_k=${LL_TOPK:-?}) or a rope encoder (${LL_ENC:-?})" >&2; exit 1
fi

# ---- horizon / step counts from the training config ------------------------
CFG_N_OBS="$(yaml_top_level n_obs_steps)"
CFG_N_ACT="$(yaml_top_level n_action_steps)"
N_OBS_STEPS="${N_OBS_STEPS:-${CFG_N_OBS:-2}}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CFG_N_ACT:-8}}"

# ---- dataset lookup from the task key --------------------------------------
resolve_dataset() {
  local key stem cand
  key="$(printf '%s' "${TASK_KEY}" | tr '[:upper:]' '[:lower:]')"   # kitchen_d1
  stem="${key%_d1}"                                                 # kitchen
  # hammercleanup -> hammer_cleanup, coffee_preperation -> coffee_preparation
  local -a stems=("${stem}" "${stem//preperation/preparation}" "${stem//hammercleanup/hammer_cleanup}")
  for s in "${stems[@]}"; do
    for cand in "${DATASET_ROOT}/${s}_d1.hdf5" "${DATASET_ROOT}/${s}.hdf5"; do
      [[ -f "${cand}" ]] && { printf '%s' "${cand}"; return 0; }
    done
  done
  cand="$(ls "${DATASET_ROOT}"/*"${stem%%_*}"*.hdf5 2>/dev/null | head -1 || true)"
  [[ -n "${cand}" ]] && { printf '%s' "${cand}"; return 0; }
  return 1
}

# ---- checkpoint discovery ---------------------------------------------------
ckpt_epoch() {  # sort key: numeric epoch if present, else -1
  local n="${1##*/}"; n="${n%.ckpt}"
  if [[ "${n}" =~ ([0-9]+)[^0-9]*$ ]]; then printf '%s' "${BASH_REMATCH[1]}"; else printf '%s' "-1"; fi
}

list_checkpoints() {  # newest (highest epoch) last
  local f
  for f in "${CKPT_DIR}"/*.ckpt; do
    [[ -e "${f}" ]] || continue
    printf '%s\t%s\n' "$(ckpt_epoch "${f}")" "$(basename "${f}")"
  done | sort -n -k1,1 | cut -f2-
}

resolve_ckpt() {  # accepts epoch_99.ckpt | epoch_99 | 99 | latest
  local want="$1" name
  case "${want}" in
    latest|last|newest) list_checkpoints | tail -1; return 0 ;;
  esac
  for name in $(list_checkpoints); do
    [[ "${name}" == "${want}" || "${name}" == "${want}.ckpt" \
       || "${name}" == "epoch_${want}.ckpt" || "${name%.ckpt}" == "${want}" ]] \
      && { printf '%s' "${name}"; return 0; }
  done
  return 1
}

resolve_hl_ckpt() {  # newest-epoch .ckpt in HL_CKPT_DIR
  local f best="" best_ep=-1 ep
  for f in "${HL_CKPT_DIR}"/*.ckpt; do
    [[ -e "${f}" ]] || continue
    ep="$(ckpt_epoch "${f}")"
    if (( ep > best_ep )); then best_ep="${ep}"; best="${f}"; fi
  done
  [[ -n "${best}" ]] && { printf '%s' "${best}"; return 0; }
  return 1
}

# ---- argument parsing -------------------------------------------------------
CKPT_REQUESTS=()
CKPT_ALL=0
CKPT_LAST_N=0
SEEDS=()
DRY_RUN=0
LIST_ONLY=0
SUMMARY_ONLY=0

usage() { awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${SELF}"; }

while (($#)); do
  case "$1" in
    -c|--checkpoint)    CKPT_REQUESTS+=("$2"); shift 2 ;;
    --checkpoints)      IFS=', ' read -r -a _cks <<<"$2"; CKPT_REQUESTS+=("${_cks[@]}"); shift 2 ;;
    -a|--all|--all-checkpoints) CKPT_ALL=1; shift ;;
    --last)             CKPT_LAST_N="$2"; shift 2 ;;
    -s|--seeds)         IFS=', ' read -r -a SEEDS <<<"$2"; shift 2 ;;
    -m|--mode)          MODE="$2"; shift 2 ;;
    -d|--dataset)       DATASET_PATH="$2"; shift 2 ;;
    --hl-ckpt)          HL_CKPT="$2"; shift 2 ;;
    -n|--episodes)      N_EPISODES="$2"; shift 2 ;;
    --max-steps)        MAX_STEPS="$2"; shift 2 ;;
    --num-envs)         NUM_ENVS="$2"; shift 2 ;;
    --n-obs-steps)      N_OBS_STEPS="$2"; shift 2 ;;
    --n-action-steps)   N_ACTION_STEPS="$2"; shift 2 ;;
    --hl-argmax-weight) HL_ARGMAX_WEIGHT="$2"; shift 2 ;;
    --videos)           SAVE_VIDEOS=1; shift ;;
    --no-videos)        SAVE_VIDEOS=0; shift ;;
    --num-video-episodes) NUM_VIDEO_EPISODES="$2"; shift 2 ;;
    -l|--list|--list-checkpoints) LIST_ONLY=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --summary-only)     SUMMARY_ONLY=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

((${#SEEDS[@]})) || SEEDS=("${SEEDS_DEFAULT[@]}")

case "${FAMILY}:${MODE}" in
  goal_gripper:parallel|goal_gripper:par)
    EVAL_SCRIPT="external/mimicgen/mimicgen/scripts/eval_ghost_high_level_parallel_2d_dit_low_level.py"
    FAMILY_TAG="GHOST"; MODE_TAG="PARALLEL" ;;
  goal_gripper:sequential|goal_gripper:seq)
    EVAL_SCRIPT="external/mimicgen/mimicgen/scripts/eval_ghost_high_level_2d_dit_low_level.py"
    FAMILY_TAG="GHOST"; MODE_TAG="SEQUENTIAL" ;;
  rope:parallel|rope:par)
    EVAL_SCRIPT="external/mimicgen/mimicgen/scripts/eval_gmm_high_level_rope_2d_dit_low_level.py"
    FAMILY_TAG="GMM_ROPE"; MODE_TAG="PARALLEL" ;;
  gmm:parallel|gmm:par)
    EVAL_SCRIPT="external/mimicgen/mimicgen/scripts/eval_gmm_high_level_parallel_2d_dit_low_level.py"
    FAMILY_TAG="GMM"; MODE_TAG="PARALLEL" ;;
  rope:*|gmm:*)
    echo "[ERROR] --mode sequential is only available for the goal_gripper family (got family=${FAMILY})" >&2; exit 2 ;;
  *) echo "[ERROR] unknown --mode '${MODE}' (expected parallel|sequential)" >&2; exit 2 ;;
esac

# ---- resolve which checkpoints to run --------------------------------------
mapfile -t AVAILABLE < <(list_checkpoints)
((${#AVAILABLE[@]})) || { echo "[ERROR] no *.ckpt files in ${CKPT_DIR}" >&2; exit 1; }

CKPTS=()
if ((CKPT_ALL)); then
  CKPTS=("${AVAILABLE[@]}")
elif ((CKPT_LAST_N > 0)); then
  mapfile -t CKPTS < <(printf '%s\n' "${AVAILABLE[@]}" | tail -n "${CKPT_LAST_N}")
elif ((${#CKPT_REQUESTS[@]})); then
  for req in "${CKPT_REQUESTS[@]}"; do
    resolved="$(resolve_ckpt "${req}" || true)"
    [[ -n "${resolved}" ]] || {
      echo "[ERROR] no checkpoint matching '${req}' in ${CKPT_DIR}. Available:" >&2
      printf '  %s\n' "${AVAILABLE[@]}" >&2
      exit 1
    }
    CKPTS+=("${resolved}")
  done
else
  CKPTS=("${AVAILABLE[-1]}")   # newest by epoch
fi

if [[ -z "${DATASET_PATH}" ]]; then
  DATASET_PATH="$(resolve_dataset || true)"
  [[ -n "${DATASET_PATH}" ]] || {
    echo "[ERROR] could not resolve a dataset for task '${TASK_KEY}' under ${DATASET_ROOT}; pass --dataset <hdf5>" >&2
    exit 1
  }
fi
if [[ -z "${HL_CKPT}" ]]; then
  HL_CKPT="$(resolve_hl_ckpt || true)"
  [[ -n "${HL_CKPT}" ]] || {
    echo "[ERROR] no *.ckpt in ${HL_CKPT_DIR}; pass --hl-ckpt <ckpt> or set HL_CKPT_DIR" >&2
    exit 1
  }
fi

ENV_PY="${INFERENCE_ROOT}/.pixi/envs/eval/bin/python"
OUTPUT_BASE="APPROACH1_${FAMILY_TAG}_${MODE_TAG}_${VARIANT_LABEL}_2D_DIT_LOW_LEVEL_${TASK_LABEL}_${N_EPISODES}_SAMPLES_DINOV2"

cat <<EOF
[derived] run dir     : ${LL_EXP_DIR}
[derived] run name    : ${RUN_CFG_NAME}
[derived] task        : ${TASK_KEY}   (label ${TASK_LABEL})
[derived] variant     : ${VARIANT_LABEL}
[derived] LL family   : ${FAMILY}   (encoder=${LL_ENC:-?} wca=${LL_WCA:-?} gmm_top_k=${LL_TOPK:-null})
[derived] mode        : ${MODE} -> ${EVAL_SCRIPT}
[derived] dataset     : ${DATASET_PATH}
[derived] HL ckpt     : ${HL_CKPT}
[derived] steps       : n_obs=${N_OBS_STEPS} n_action=${N_ACTION_STEPS}  max_steps=${MAX_STEPS}
[derived] parallel    : num_envs=${NUM_ENVS} dtype=${INFERENCE_DTYPE} videos=${SAVE_VIDEOS} (first ${NUM_VIDEO_EPISODES})
[derived] seeds       : ${SEEDS[*]}
[derived] checkpoints : ${CKPTS[*]}
[derived] available   : ${AVAILABLE[*]}
[derived] output base : ${OUTPUT_BASE}
EOF
((LIST_ONLY)) && exit 0

for f in "${DATASET_PATH}" "${HL_CKPT}" \
         "${HYDRA_CFG}" \
         "${LFD3D_REPO}/src/lfd3d/models/articubot.py" \
         "${DIT_2D_REPO}/diffusion_policy/policy/flow_matching_dit_image_policy.py" \
         "${INFERENCE_ROOT}/eval_smith_utils.py" \
         "${INFERENCE_ROOT}/equi_diffpo/gym_util/async_vector_env.py" \
         "${INFERENCE_ROOT}/external/robomimic/robomimic/envs/env_robosuite.py" \
         "${INFERENCE_ROOT}/${EVAL_SCRIPT}" \
         "${INFERENCE_ROOT}/shell_scripts/approach2_eval_utils.sh" \
         "${INFERENCE_ROOT}/scripts/summarize_approach2_evals.py" \
         "${ENV_PY}"; do
  [[ -e "${f}" ]] || { echo "[ERROR] missing required path: ${f}" >&2; exit 1; }
done
for ckpt in "${CKPTS[@]}"; do
  [[ -s "${CKPT_DIR}/${ckpt}" ]] || { echo "[ERROR] missing/empty checkpoint: ${CKPT_DIR}/${ckpt}" >&2; exit 1; }
done
if [[ -n "${TEXT_EMBED_CACHE}" && ! -e "${TEXT_EMBED_CACHE}" ]]; then
  echo "[ERROR] TEXT_EMBED_CACHE set but missing: ${TEXT_EMBED_CACHE}" >&2; exit 1
fi

cd "${INFERENCE_ROOT}"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DISPLAY=${DISPLAY:-:99}
export PYTHONPATH="${LFD3D_REPO}/src:${DIT_2D_REPO}:${INFERENCE_ROOT}:${PYTHONPATH:-}"
# mujoco_py (legacy binding, used by robomimic's EnvRobosuite) needs the
# standalone MuJoCo 2.1 native lib + nvidia GL libs on LD_LIBRARY_PATH, and
# `patchelf` (from the eval env) on PATH for its one-time Cython build.
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export PATH="${INFERENCE_ROOT}/.pixi/envs/eval/bin:${PATH}"

# --------------------------------------------------------------------------- #
# do_merge SRC DST ORIG_SEED -- folds a sibling _RESUME_<N> dir into the main dir.
# (Same helper as the *_ALL_SEEDS.sh scripts.)
# --------------------------------------------------------------------------- #
do_merge() {
  local SRC="$1" DST="$2" THE_ORIG_SEED="$3"
  if [[ ! -f "${SRC}/results.jsonl" ]]; then
    echo "[merge] ${SRC} has no results.jsonl, skipping"
    return 0
  fi
  mkdir -p "${DST}/media"
  "${ENV_PY}" - "${SRC}" "${DST}" "${THE_ORIG_SEED}" <<'PYEOF'
import json, sys, shutil
from pathlib import Path
src_dir = Path(sys.argv[1]); dst_dir = Path(sys.argv[2]); orig_seed = int(sys.argv[3])
n = 0
with open(src_dir / "results.jsonl") as fi, open(dst_dir / "results.jsonl", "a") as fo:
    for line in fi:
        if not line.strip():
            continue
        d = json.loads(line); n += 1
        seed = d["seed"]; new_ep = seed - orig_seed + 1
        outcome = "success" if d["success"] else "failure"
        d["episode"] = new_ep
        for key, sub in (("video", "media"), ("video_with_goal_overlay", "media_with_goal_overlay")):
            old = d.get(key)
            if not old:
                continue
            candidates = [Path(old), src_dir / sub / Path(old).name]
            old_path = next((p for p in candidates if p.exists()), None)
            new_name = f"episode_{new_ep:03d}_seed_{seed}_{outcome}.mp4"
            (dst_dir / sub).mkdir(parents=True, exist_ok=True)
            if old_path is not None:
                shutil.move(str(old_path), str(dst_dir / sub / new_name))
            else:
                print(f"[merge][WARN] missing source video for line {n}: {old}")
            d[key] = str(dst_dir / sub / new_name)
        fo.write(json.dumps(d) + "\n")
print(f"[merge] appended {n} rows from {src_dir} -> {dst_dir}/results.jsonl")
PYEOF
  touch "${SRC}/.merged"
}

run_one_seed() {
  local ckpt_name="$1" ckpt_tag="$2" seed="$3"
  local tag="${ckpt_tag}_${seed}_SEED"
  local MAIN_OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${tag}"

  echo
  echo "==========================================================================="
  echo "[${tag}] CKPT=${ckpt_name}  SEED=${seed}  OUTPUT=${MAIN_OUTPUT_DIR}"
  echo "==========================================================================="

  # ---- resume bookkeeping -------------------------------------------------
  if ((!DRY_RUN)); then
    shopt -s nullglob
    for resume_dir in "${MAIN_OUTPUT_DIR}_RESUME_"*; do
      if [[ -d "${resume_dir}" && ! -f "${resume_dir}/.merged" ]]; then
        echo "[merge] folding $(basename "${resume_dir}") into ${MAIN_OUTPUT_DIR}"
        do_merge "${resume_dir}" "${MAIN_OUTPUT_DIR}" "${seed}"
      fi
    done
    shopt -u nullglob
  fi
  local COMPLETED=0
  if [[ -f "${MAIN_OUTPUT_DIR}/results.jsonl" ]]; then
    COMPLETED=$(grep -c . "${MAIN_OUTPUT_DIR}/results.jsonl" || true)
  fi
  if (( COMPLETED >= N_EPISODES )); then
    echo "[resume] all ${N_EPISODES} episodes already in ${MAIN_OUTPUT_DIR}. Skipping."
    return 0
  fi
  local PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}" CUR_SEED="${seed}" CUR_N_EP="${N_EPISODES}"
  if (( COMPLETED > 0 )); then
    CUR_SEED=$(( seed + COMPLETED ))
    CUR_N_EP=$(( N_EPISODES - COMPLETED ))
    PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}_RESUME_${COMPLETED}"
    echo "[resume] ${COMPLETED} episodes already done; running ${CUR_N_EP} more (seeds ${CUR_SEED}..$(( CUR_SEED + CUR_N_EP - 1 )))"
    echo "[resume] Python writes to ${PY_OUTPUT_DIR}; folded into ${MAIN_OUTPUT_DIR} on success."
  fi

  export TMPDIR="${TMPDIR_BASE:-/tmp}/approach1_${FAMILY_TAG,,}_${MODE_TAG,,}_${VARIANT_LABEL,,}_eval_${tag}_$$"
  mkdir -p "${TMPDIR}"

  local VIDEO_FLAG
  if [[ "${SAVE_VIDEOS}" == "0" ]]; then
    VIDEO_FLAG=(--no-save_videos)
  else
    VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
  fi
  local ROBOSUITE_FLAG=() TEXT_EMBED_FLAG=() MODE_FLAGS=() FAMILY_FLAGS=()
  [[ -n "${ROBOSUITE_ROOT}" ]] && ROBOSUITE_FLAG=(--robosuite_root "${ROBOSUITE_ROOT}")
  [[ -n "${TEXT_EMBED_CACHE}" ]] && TEXT_EMBED_FLAG=(--text_embed_cache "${TEXT_EMBED_CACHE}")
  if [[ "${MODE_TAG}" == "PARALLEL" ]]; then
    MODE_FLAGS=(--num_envs "${NUM_ENVS}" --inference_dtype "${INFERENCE_DTYPE}"
                --num_video_episodes "${NUM_VIDEO_EPISODES}")
  else
    MODE_FLAGS=(--no-save_goal_overlay_videos)
  fi
  # Only the GHOST evaluators take --hl_argmax_weight; the GMM ones feed the
  # full distribution to the LL and have no sampling step.
  if [[ "${FAMILY}" == "goal_gripper" ]]; then
    FAMILY_FLAGS=(--hl_argmax_weight "${HL_ARGMAX_WEIGHT}")
  fi

  local -a cmd=(
    "${ENV_PY}" "${EVAL_SCRIPT}"
      --dataset_path         "${DATASET_PATH}"
      --high_level_ckpt      "${HL_CKPT}"
      --low_level_exp_dir    "${LL_EXP_DIR}"
      --low_level_checkpoint "${ckpt_name}"
      --lfd3d_repo           "${LFD3D_REPO}"
      --dit_2d_repo          "${DIT_2D_REPO}"
      "${ROBOSUITE_FLAG[@]}"
      "${TEXT_EMBED_FLAG[@]}"
      --hl_in_channels       "${HL_IN_CHANNELS}"
      "${FAMILY_FLAGS[@]}"
      --n_episodes           "${CUR_N_EP}"
      --max_steps            "${MAX_STEPS}"
      --seed                 "${CUR_SEED}"
      --n_obs_steps          "${N_OBS_STEPS}"
      --n_action_steps       "${N_ACTION_STEPS}"
      --camera_h             "${CAMERA_H}"
      --camera_w             "${CAMERA_W}"
      "${MODE_FLAGS[@]}"
      "${VIDEO_FLAG[@]}"
      --output_dir           "${PY_OUTPUT_DIR}"
  )
  if ((DRY_RUN)); then printf '[dry-run]'; printf ' %q' "${cmd[@]}"; echo; return 0; fi
  "${cmd[@]}"

  if [[ "${PY_OUTPUT_DIR}" != "${MAIN_OUTPUT_DIR}" ]]; then
    echo "[merge] folding ${PY_OUTPUT_DIR} into ${MAIN_OUTPUT_DIR}"
    do_merge "${PY_OUTPUT_DIR}" "${MAIN_OUTPUT_DIR}" "${seed}"
  fi
}

TMPDIR_BASE="${TMPDIR:-/tmp}"
# Keep the real terminal fds so each checkpoint re-logs to its own eval.log
# instead of nesting tee pipelines from previous iterations.
exec {ORIG_STDOUT}>&1 {ORIG_STDERR}>&2
# The layout helpers are Approach-agnostic (evaluations/<ckpt>/<eval_name>/...).
source "${INFERENCE_ROOT}/shell_scripts/approach2_eval_utils.sh"

for ckpt in "${CKPTS[@]}"; do
  CKPT_TAG="$(printf '%s' "${ckpt%.ckpt}" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9._-' '_')"
  CKPT_TAG="${CKPT_TAG%_}"

  # Fresh output layout per checkpoint: evaluations/<ckpt>/<eval_name>/...
  if ((DRY_RUN)); then
    SCRIPT_DIR="${LL_EXP_DIR}/evaluations/${ckpt%.ckpt}/${OUTPUT_BASE}"
  else
    exec >&${ORIG_STDOUT} 2>&${ORIG_STDERR}
    approach2_prepare_eval_layout "${LL_EXP_DIR}" "${ckpt}" "${SELF}" "${OUTPUT_BASE}"
    approach2_start_eval_logging
  fi

  if ((SUMMARY_ONLY)); then
    echo "[summary-only] rebuilding summaries from existing results in ${SCRIPT_DIR}"
  else
    for seed in "${SEEDS[@]}"; do
      run_one_seed "${ckpt}" "${CKPT_TAG}" "${seed}"
    done
  fi

  echo
  echo "Checkpoint ${ckpt}: ${#SEEDS[@]} seed(s) done. Outputs:"
  for seed in "${SEEDS[@]}"; do
    echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_${CKPT_TAG}_${seed}_SEED"
  done
  if ((DRY_RUN)); then continue; fi

  # Per-eval summary (evaluations/<epoch>/<eval_name>/summary_all_seeds.*)
  approach2_write_combined_summary "${SEEDS[@]}"

  # Same combined-seed summary one level up, in the epoch folder itself.
  epoch_dir="${LL_EXP_DIR}/evaluations/${ckpt%.ckpt}"
  "${ENV_PY}" "${INFERENCE_ROOT}/scripts/summarize_approach2_evals.py" \
      --eval-dir       "${APPROACH2_EVAL_ROOT}" \
      --checkpoint     "${APPROACH2_CHECKPOINT_TAG}" \
      --expected-seeds "${SEEDS[@]}" \
      --json-output    "${epoch_dir}/summary_all_seeds.json" \
      --text-output    "${epoch_dir}/summary_all_seeds.txt" >/dev/null
  echo "Epoch-level summary: ${epoch_dir}/summary_all_seeds.json"
done

echo
echo "All done: ${#CKPTS[@]} checkpoint(s) x ${#SEEDS[@]} seed(s)."
