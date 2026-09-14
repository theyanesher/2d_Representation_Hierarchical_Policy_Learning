#!/usr/bin/env bash
# =============================================================================
# Sweep eval of the PushT Approach 2 checkpoints.
#
# Drives the per-variant harnesses in this directory
# (BAYESIAN_ACC_JERK/, RDP/, FIXED_INTERVAL/, BAYESIAN_VELOCITY/, RANDOM/ --
# all five share an identical CLI and differ only in replanning strategy) over
# every run dir found under CKPT_ROOT.
#
# Sweep axes:  experiment dir  x  variant  x  checkpoint epoch  x  seed
#
# Outputs land in a tree that mirrors the sweep:
#   ${OUTPUT_ROOT}/<exp_dir>/<VARIANT>/<epoch_NNN>/seed_<seed>/
#       args.json  results.jsonl  summary.json  media/*.mp4
# and an aggregate ${OUTPUT_ROOT}/sweep_summary.csv is (re)written at the end.
#
# Resume: a run whose results.jsonl already holds >= N_EPISODES rows is
# skipped, so re-launching after an interrupt continues where it stopped.
#
# ---------------------------------------------------------------------------
# IMPORTANT -- PYTHONPATH
# ---------------------------------------------------------------------------
# eval_approach2_pusht.py does its OWN sys.path bootstrap and requires the
# Low_Level repo's `diffusion_policy` NAMESPACE package. The PushT repo root
# must never reach sys.path, or its own `diffusion_policy` shadows the
# Low_Level one and FlowMatchingDiTGoalGMMPolicy appears to be missing.
#
# The PushT pixi env sets PYTHONPATH=<repo root> on activation (so plain
# `pixi run python train.py` works), which is exactly what must NOT be set
# here. Every invocation below therefore REPLACES PYTHONPATH outright with
# just OVERLAY (never appends to an inherited one).
#
# OVERLAY holds the four sim packages the ArticuBot training env lacks --
# pygame, pymunk, shapely, av -- installed with --no-deps so they cannot drag
# in a numpy 2.x that would shadow the env's pinned numpy 1.23. It contains no
# diffusion_policy, so it cannot shadow the Low_Level namespace package.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------------------------------------------------------- #
# Defaults (override with the flags below)
# --------------------------------------------------------------------------- #
CKPT_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning/pusht_appraoch2_ckpts"
# NOT <main repo>/Low_Level_and_Inference -- that tree is a __pycache__-only
# skeleton (0 .py files, no env/ at all, and no .pyc for the GMM policy
# either), so it cannot supply the namespace package.
#
# Two complete copies exist locally; they are NOT identical:
#   this default  /home/theyanesh/worktrees/theya_low_level_dit2d/...
#                 branch theya_low_level, policy 18103 B, mtime Aug 27
#   alternative   /home/theyanesh/Pratik_Low_Level/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference
#                 branch GMM_Based_Training_Low_Level, policy 12361 B, Aug 17
# The default is the newer one, which predates the Aug 29 checkpoints by two
# days. If a run dies on a state-dict key mismatch, try --ll_repo with the
# other path before touching anything else.
LL_REPO="/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference"
# Interpreter = the Low_Level repo's own pixi env (the ArticuBot workspace the
# checkpoints were TRAINED in). It is the only local env with the deps the
# Approach 2 policy pulls in: the checkpoint's _target_ is
# train_diffusion_unet_hybrid_workspace, whose import chain reaches robomimic
# (0.2.0 here), plus transformers 4.56 / timm / xformers for the DINOv2 trunk.
# The PushT pixi env in this repo deliberately does NOT carry that stack -- it
# covers the stock diffusion_policy PushT code, not Approach 2.
# Used directly rather than via `pixi run` so a stale lock can never trigger a
# multi-GB re-solve mid-sweep. Its site-packages has no diffusion_policy, so
# nothing shadows the Low_Level namespace package.
ENV_PY="/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference/.pixi/envs/default/bin/python"
OUTPUT_ROOT="${SCRIPT_DIR}/SWEEP_RESULTS"
# Kept outside every repo and outside .pixi/, so `pixi install` in the
# Low_Level workspace can never delete it.
OVERLAY="${HOME}/.cache/pusht_approach2_sim_overlay"
AUTO_SETUP=1
GPU=""                             # e.g. 1 -> CUDA_VISIBLE_DEVICES=1

EXP_FILTER=""                      # substring; empty = every run dir
VARIANTS="BAYESIAN_ACC_JERK"       # or: all, or a comma list
EPOCHS="epoch_199.ckpt"            # or: all, or a comma list
SEEDS="100000"                     # comma list, e.g. 100000,150000,250000

N_EPISODES=50
MAX_STEPS=300
N_OBS_STEPS=2
N_ACTION_STEPS=8
RENDER_SIZE=256
OBS_HISTORY=rolling
ACTION_START=0
SAVE_VIDEOS=1
VIDEO_FPS=10
DRY_RUN=0

ALL_VARIANTS="BAYESIAN_ACC_JERK RDP FIXED_INTERVAL BAYESIAN_VELOCITY RANDOM"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<EOF

Usage: $(basename "$0") [options]

  --ckpt_root DIR     Root holding the run dirs      (default: ${CKPT_ROOT})
  --exp_filter STR    Only run dirs whose name contains STR (e.g. uvd)
  --ll_repo DIR       Low_Level_and_Inference root   (default: ${LL_REPO})
  --env_python PATH   Interpreter to run the harness with
                      (default: the Low_Level repo pixi env python)
  --overlay DIR       Dir holding pygame/pymunk/shapely/av
                      (default: ${OVERLAY})
  --no_auto_setup     Fail instead of pip-installing a missing overlay
  --gpu N             Pin to one GPU (sets CUDA_VISIBLE_DEVICES); 'none' forces
                      CPU. Both cards on this box are often saturated by other
                      jobs -- check nvidia-smi first; the policy needs ~2GB.
  --output_root DIR   Where results go               (default: ${OUTPUT_ROOT})
  --variants LIST     'all' or comma list            (default: ${VARIANTS})
  --epochs LIST       'all' or comma list of ckpts   (default: ${EPOCHS})
  --seeds LIST        comma list                     (default: ${SEEDS})
  --n_episodes N      episodes per run               (default: ${N_EPISODES})
  --max_steps N                                      (default: ${MAX_STEPS})
  --n_obs_steps N                                    (default: ${N_OBS_STEPS})
  --n_action_steps N                                 (default: ${N_ACTION_STEPS})
  --action_start N                                   (default: ${ACTION_START})
  --obs_history M     rolling | repeat_current       (default: ${OBS_HISTORY})
  --render_size N                                    (default: ${RENDER_SIZE})
  --no_videos         disable mp4 writing
  --video_fps N                                      (default: ${VIDEO_FPS})
  --dry_run           print the planned runs and exit
  -h, --help          this message

Examples:
  # everything: all variants, all epochs, 3 seeds
  $(basename "$0") --variants all --epochs all --seeds 100000,150000,250000

  # quick smoke: 2 episodes, no video, final epoch only
  $(basename "$0") --n_episodes 2 --no_videos --dry_run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_root)      CKPT_ROOT="$2"; shift 2 ;;
    --exp_filter)     EXP_FILTER="$2"; shift 2 ;;
    --ll_repo)        LL_REPO="$2"; shift 2 ;;
    --env_python)     ENV_PY="$2"; shift 2 ;;
    --overlay)        OVERLAY="$2"; shift 2 ;;
    --no_auto_setup)  AUTO_SETUP=0; shift ;;
    --gpu)            GPU="$2"; shift 2 ;;
    --output_root)    OUTPUT_ROOT="$2"; shift 2 ;;
    --variants)       VARIANTS="$2"; shift 2 ;;
    --epochs)         EPOCHS="$2"; shift 2 ;;
    --seeds)          SEEDS="$2"; shift 2 ;;
    --n_episodes)     N_EPISODES="$2"; shift 2 ;;
    --max_steps)      MAX_STEPS="$2"; shift 2 ;;
    --n_obs_steps)    N_OBS_STEPS="$2"; shift 2 ;;
    --n_action_steps) N_ACTION_STEPS="$2"; shift 2 ;;
    --action_start)   ACTION_START="$2"; shift 2 ;;
    --obs_history)    OBS_HISTORY="$2"; shift 2 ;;
    --render_size)    RENDER_SIZE="$2"; shift 2 ;;
    --no_videos)      SAVE_VIDEOS=0; shift ;;
    --video_fps)      VIDEO_FPS="$2"; shift 2 ;;
    --dry_run)        DRY_RUN=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------- #
# Resolve sweep axes
# --------------------------------------------------------------------------- #
if [[ "${VARIANTS}" == "all" ]]; then
  VARIANT_LIST=(${ALL_VARIANTS})
else
  IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"
fi
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"

# Experiment dirs = immediate children of CKPT_ROOT that have checkpoints/.
# -L because pusht_appraoch2_ckpts is a symlink into /data.
EXP_DIRS=()
while IFS= read -r d; do
  [[ -d "${d}/checkpoints" ]] || continue
  if [[ -n "${EXP_FILTER}" && "$(basename "${d}")" != *"${EXP_FILTER}"* ]]; then
    continue
  fi
  EXP_DIRS+=("${d}")
done < <(find -L "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)

# --------------------------------------------------------------------------- #
# Preflight -- fail loudly before burning GPU hours
# --------------------------------------------------------------------------- #
fail() { echo "[ERROR] $*" >&2; exit 1; }

[[ ${#EXP_DIRS[@]} -gt 0 ]] || fail "no run dirs with checkpoints/ under ${CKPT_ROOT}"
[[ -x "${ENV_PY}" ]]         || fail "interpreter not found/executable: ${ENV_PY}"
[[ -d "${LL_REPO}" ]]        || fail "Low_Level repo not found: ${LL_REPO}"

# The two modules the harness resolves out of the Low_Level namespace package.
# Their absence is the failure mode that otherwise shows up deep in a rollout.
LL_POLICY="${LL_REPO}/diffusion_policy/diffusion_policy/policy/flow_matching_dit_goal_gmm_policy.py"
LL_ENV="${LL_REPO}/diffusion_policy/diffusion_policy/env/pusht/pusht_image_env.py"
for f in "${LL_POLICY}" "${LL_ENV}"; do
  [[ -f "${f}" ]] || fail "missing required Low_Level module: ${f}
        --low_level_repo must point at a Low_Level_and_Inference checkout whose
        diffusion_policy/diffusion_policy/{policy,env/pusht}/ hold real .py
        sources (a __pycache__-only tree will not import)."
done

for v in "${VARIANT_LIST[@]}"; do
  [[ -f "${SCRIPT_DIR}/${v}/eval_approach2_pusht.py" ]] \
    || fail "no harness for variant '${v}': ${SCRIPT_DIR}/${v}/eval_approach2_pusht.py"
done

# A diffusion_policy in the interpreter's site-packages would shadow the
# Low_Level NAMESPACE package and make the Approach 2 policy class "vanish".
ENV_SP="$("${ENV_PY}" -c 'import site; print(site.getsitepackages()[0])')"
[[ -e "${ENV_SP}/diffusion_policy" ]] && fail "${ENV_SP}/diffusion_policy exists -- it would
        shadow the Low_Level namespace package. Rename it out of the way."
"${ENV_PY}" -c 'import robomimic' 2>/dev/null \
  || fail "robomimic not importable in ${ENV_PY}
        The Approach 2 checkpoint's workspace target imports it. Point
        --env_python at the Low_Level/ArticuBot pixi env."

# The ArticuBot env has robomimic/transformers/timm but NOT the PushT sim
# stack. Provision the four missing packages into OVERLAY once.
# pymunk MUST stay <7: Space.add_collision_handler, which PushTEnv._setup
# calls, was removed in pymunk 7. 6.10 is verified working.
SIM_PKGS=(pygame "pymunk<7" shapely av)
overlay_ok() {
  env PYTHONPATH="${OVERLAY}" "${ENV_PY}" -c \
    'import pygame, pymunk, shapely, av' >/dev/null 2>&1
}
if ! overlay_ok; then
  if [[ "${AUTO_SETUP}" != "1" ]]; then
    fail "overlay incomplete at ${OVERLAY} and --no_auto_setup was given.
        Provision it with:
          env -u PYTHONPATH ${ENV_PY} -m pip install --target ${OVERLAY} \\
              --no-deps ${SIM_PKGS[*]}"
  fi
  echo "[setup] provisioning sim overlay -> ${OVERLAY}"
  mkdir -p "${OVERLAY}"
  # --no-deps is load-bearing: shapely/av would otherwise pull numpy 2.x into
  # the overlay, where it would shadow the env's numpy 1.23 and break torch.
  env -u PYTHONPATH "${ENV_PY}" -m pip install --target "${OVERLAY}" \
      --no-deps "${SIM_PKGS[@]}" \
    || fail "could not provision overlay at ${OVERLAY}"
  overlay_ok || fail "overlay still incomplete after install: ${OVERLAY}"
  echo "[setup] overlay ready."
fi

# --------------------------------------------------------------------------- #
# Build the run plan
# --------------------------------------------------------------------------- #
declare -a PLAN_EXP PLAN_VAR PLAN_CKPT PLAN_SEED PLAN_OUT

for exp in "${EXP_DIRS[@]}"; do
  exp_name="$(basename "${exp}")"

  if [[ "${EPOCHS}" == "all" ]]; then
    mapfile -t CKPT_LIST < <(find -L "${exp}/checkpoints" -maxdepth 1 -name '*.ckpt' -printf '%f\n' \
                             | sort -t_ -k2 -n)
  else
    IFS=',' read -r -a CKPT_LIST <<< "${EPOCHS}"
  fi

  for v in "${VARIANT_LIST[@]}"; do
    for ck in "${CKPT_LIST[@]}"; do
      [[ -f "${exp}/checkpoints/${ck}" ]] || fail "missing checkpoint: ${exp}/checkpoints/${ck}"
      for s in "${SEED_LIST[@]}"; do
        PLAN_EXP+=("${exp}")
        PLAN_VAR+=("${v}")
        PLAN_CKPT+=("${ck}")
        PLAN_SEED+=("${s}")
        PLAN_OUT+=("${OUTPUT_ROOT}/${exp_name}/${v}/${ck%.ckpt}/seed_${s}")
      done
    done
  done
done

TOTAL=${#PLAN_EXP[@]}

echo "==========================================================================="
echo " PushT Approach 2 checkpoint sweep"
echo "==========================================================================="
echo " ckpt root   : ${CKPT_ROOT}"
echo " run dirs    : ${#EXP_DIRS[@]}"
for e in "${EXP_DIRS[@]}"; do echo "                - $(basename "${e}")"; done
echo " variants    : ${VARIANT_LIST[*]}"
echo " seeds       : ${SEED_LIST[*]}"
echo " episodes    : ${N_EPISODES}  (max_steps ${MAX_STEPS})"
echo " LL repo     : ${LL_REPO}"
echo " interpreter : ${ENV_PY}"
echo " overlay     : ${OVERLAY}"
echo " output root : ${OUTPUT_ROOT}"
echo " total runs  : ${TOTAL}"
echo "==========================================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  for ((i=0; i<TOTAL; i++)); do
    printf '  [%3d/%3d] %-28s %-20s %-12s seed=%s\n' \
      "$((i+1))" "${TOTAL}" \
      "$(basename "${PLAN_EXP[$i]}" | cut -c1-28)" \
      "${PLAN_VAR[$i]}" "${PLAN_CKPT[$i]}" "${PLAN_SEED[$i]}"
  done
  echo "[dry run] nothing executed."
  exit 0
fi

# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"   # headless pygame
export PYTHONNOUSERSITE=1
if [[ "${GPU}" == "none" ]]; then
  # The harness picks its device via torch.cuda.is_available(), so hiding every
  # GPU is enough to force CPU -- no edit to eval_approach2_pusht.py needed.
  export CUDA_VISIBLE_DEVICES=""
  echo "[gpu] CPU only (CUDA_VISIBLE_DEVICES emptied) -- the 180M-param DiT is SLOW here"
elif [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  echo "[gpu] CUDA_VISIBLE_DEVICES=${GPU}"
fi
export TMPDIR="${TMPDIR:-/tmp/pusht_approach2_sweep_$$}"
mkdir -p "${TMPDIR}" "${OUTPUT_ROOT}"

VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
[[ "${SAVE_VIDEOS}" == "0" ]] && VIDEO_FLAG=(--no-save_videos)

N_OK=0; N_SKIP=0; N_FAIL=0
declare -a FAILED

for ((i=0; i<TOTAL; i++)); do
  exp="${PLAN_EXP[$i]}"; v="${PLAN_VAR[$i]}"
  ck="${PLAN_CKPT[$i]}"; s="${PLAN_SEED[$i]}"; out="${PLAN_OUT[$i]}"

  echo
  echo "---------------------------------------------------------------------------"
  echo "[$((i+1))/${TOTAL}] $(basename "${exp}")"
  echo "          variant=${v}  ckpt=${ck}  seed=${s}"
  echo "          -> ${out}"
  echo "---------------------------------------------------------------------------"

  done_n=0
  [[ -f "${out}/results.jsonl" ]] && done_n=$(wc -l < "${out}/results.jsonl")
  if (( done_n >= N_EPISODES )); then
    echo "[skip] ${done_n}/${N_EPISODES} episodes already present."
    N_SKIP=$((N_SKIP+1))
    continue
  fi
  (( done_n > 0 )) && echo "[note] ${done_n} partial episodes present; rerunning this cell from scratch."

  mkdir -p "${out}"

  # `env -u PYTHONPATH`: see the PYTHONPATH banner at the top of this file.
  if env PYTHONPATH="${OVERLAY}" "${ENV_PY}" "${SCRIPT_DIR}/${v}/eval_approach2_pusht.py" \
          --low_level_exp_dir    "${exp}"            \
          --low_level_checkpoint "${ck}"             \
          --low_level_repo       "${LL_REPO}"        \
          --n_episodes           "${N_EPISODES}"     \
          --max_steps            "${MAX_STEPS}"      \
          --seed                 "${s}"              \
          --n_obs_steps          "${N_OBS_STEPS}"    \
          --n_action_steps       "${N_ACTION_STEPS}" \
          --action_start         "${ACTION_START}"   \
          --obs_history          "${OBS_HISTORY}"    \
          --render_size          "${RENDER_SIZE}"    \
          "${VIDEO_FLAG[@]}"                         \
          --output_dir           "${out}"
  then
    N_OK=$((N_OK+1))
  else
    echo "[FAIL] $(basename "${exp}") / ${v} / ${ck} / seed ${s}" >&2
    FAILED+=("$(basename "${exp}")|${v}|${ck}|${s}")
    N_FAIL=$((N_FAIL+1))
  fi
done

# --------------------------------------------------------------------------- #
# Aggregate every summary.json under OUTPUT_ROOT into one table + CSV
# --------------------------------------------------------------------------- #
echo
echo "==========================================================================="
echo " Sweep finished:  ${N_OK} ok, ${N_SKIP} skipped, ${N_FAIL} failed  (of ${TOTAL})"
echo "==========================================================================="

if (( N_FAIL > 0 )); then
  echo "Failed cells:"
  for f in "${FAILED[@]}"; do echo "  ${f//|/  }"; done
  echo
fi

env PYTHONPATH="${OVERLAY}" "${ENV_PY}" - "${OUTPUT_ROOT}" <<'PYEOF'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for sm in sorted(root.rglob("summary.json")):
    try:
        d = json.loads(sm.read_text())
    except Exception as e:
        print(f"[warn] unreadable {sm}: {e}")
        continue
    rel = sm.parent.relative_to(root).parts
    # <exp>/<variant>/<epoch>/seed_<n>
    exp, variant, epoch, seed = (list(rel) + ["?"] * 4)[:4]
    rows.append({
        "experiment": exp, "variant": variant, "epoch": epoch,
        "seed": seed.replace("seed_", ""),
        "n_episodes": d.get("n_episodes", ""),
        "success_rate": d.get("success_rate", ""),
        "successes": d.get("successes", ""),
        "mean_coverage": d.get("mean_coverage", ""),
        "std_coverage": d.get("std_coverage", ""),
        "mean_reward": d.get("mean_reward", ""),
    })

if not rows:
    print("No summary.json found yet under", root)
    sys.exit(0)

out_csv = root / "sweep_summary.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

def s(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)

w_exp = max(10, min(46, max(len(r["experiment"]) for r in rows)))
hdr = f'{"experiment":<{w_exp}}  {"variant":<18} {"epoch":<11} {"seed":<7} {"succ%":>7} {"coverage":>10}'
print()
print(hdr)
print("-" * len(hdr))
for r in rows:
    sr = r["success_rate"]
    sr_txt = f"{sr*100:6.1f}%" if isinstance(sr, (int, float)) else f"{str(sr):>7}"
    print(f'{r["experiment"][:w_exp]:<{w_exp}}  {r["variant"]:<18} '
          f'{r["epoch"]:<11} {r["seed"]:<7} {sr_txt} {s(r["mean_coverage"]):>10}')
print()
print("CSV:", out_csv)
PYEOF

echo
echo "Results tree: ${OUTPUT_ROOT}"
exit $(( N_FAIL > 0 ? 1 : 0 ))
