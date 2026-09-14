#!/usr/bin/env bash
# =============================================================================
# Approach 2 eval -- PushT, RDP variant, with optional validation against an
# older reference run.
#
# Runs the RDP harness through eval_pusht_approach2_ckpts.sh, then (with
# --compare) diffs the fresh results.jsonl against a reference one seed-by-seed
# via compare_results.py, so you can confirm this machine reproduces numbers
# obtained earlier elsewhere.
#
# Results:  ./RESULTS_RDP/<exp>/RDP/<epoch>/seed_<seed>/
#
# ---------------------------------------------------------------------------
# About the reference that ships in this repo
# ---------------------------------------------------------------------------
# BAYESIAN_ACC_JERK/BACKUP/results.jsonl is the only reference checked in here,
# and it CANNOT be reproduced on this machine as-is:
#   * it is a BAYESIAN_ACC_JERK run, not RDP;
#   * it is partial -- 18 episodes of the intended 50;
#   * it used checkpoint epoch_99.ckpt from run dir
#     2026.08.21/02.24.45_push_t_task_APPROACH2_c1_0.1_206demo..., which is a
#     PSC path. Neither that run dir nor any epoch_99.ckpt exists locally --
#     the local ckpt root only has the Aug 29 mix_bspline and uvd runs, at
#     epochs 20..199.
# So point --compare at your own older RDP results.jsonl (copied over from
# PSC), or first copy that Aug 21 run dir across and use --ckpt_root/--epochs
# to target it.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   ./eval_pusht_rdp.sh --gpu 1                          # both run dirs, RDP
#   ./eval_pusht_rdp.sh --gpu 1 --exp uvd                # one run dir
#   ./eval_pusht_rdp.sh --gpu 1 --exp uvd --compare /path/old/results.jsonl
#   ./eval_pusht_rdp.sh --gpu 1 --exp uvd --self_check   # determinism: run twice, diff
#
# Any other flag is forwarded to the sweep driver (--epochs, --seeds,
# --n_episodes, --ckpt_root, --ll_repo, --dry_run, ...).
#
# The eval defaults are already the ones the older runs used -- n_episodes 50,
# max_steps 300, seed 100000, n_obs_steps 2, n_action_steps 8, render_size 256,
# legacy_env on -- so a comparison run needs no extra flags to line up.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/eval_pusht_approach2_ckpts.sh"
COMPARE="${SCRIPT_DIR}/compare_results.py"
OUTPUT_ROOT="${SCRIPT_DIR}/RESULTS_RDP"

EXP="both"
REF=""
SELF_CHECK=0
ALLOW_NON_RDP=0
TOL="1e-6"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)        EXP="$2"; shift 2 ;;
    --compare)    REF="$2"; shift 2 ;;
    --tol)        TOL="$2"; shift 2 ;;
    --self_check) SELF_CHECK=1; shift ;;
    --allow_non_rdp) ALLOW_NON_RDP=1; shift ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

fail() { echo "[ERROR] $*" >&2; exit 1; }

[[ -x "${DRIVER}" ]]  || fail "driver not executable: ${DRIVER}"
[[ -f "${COMPARE}" ]] || fail "compare tool missing: ${COMPARE}"

case "${EXP}" in
  mix|mix_bspline) FILTER="mix_bspline" ;;
  uvd)             FILTER="uvd" ;;
  both|all)        FILTER="" ;;
  *) fail "--exp must be one of: mix, uvd, both (got '${EXP}')" ;;
esac

# Comparing needs exactly one run dir, or it is ambiguous which results.jsonl
# to diff.
if [[ -n "${REF}" || "${SELF_CHECK}" == "1" ]]; then
  [[ -n "${FILTER}" ]] || fail "--compare/--self_check need a single run dir; pass --exp mix or --exp uvd"
fi
if [[ -n "${REF}" ]]; then
  [[ -f "${REF}" ]] || fail "reference results.jsonl not found: ${REF}"
fi

FILTER_ARGS=()
[[ -n "${FILTER}" ]] && FILTER_ARGS=(--exp_filter "${FILTER}")

# --------------------------------------------------------------------------- #
# Guard: is there actually an RDP-TRAINED checkpoint to evaluate?
#
# RDP is a separate training run, not an inference mode. All five
# */eval_approach2_pusht.py harnesses in this directory are byte-identical
# (md5 32772cf8126110cc93b40fb79cf682d6); the per-variant PSC wrappers differ
# only in LL_EXP_DIR -- i.e. which trained checkpoint they point at:
#     BAYESIAN_ACC_JERK -> .../11.56.53_push_t_task_APPROACH2_c1_0.1_206demo...
#     RDP               -> .../11.56.53_push_t_task_APPROACH2_RDP_c1_0.1_...rdp
#
# So writing results into an "RDP/" directory from a non-RDP checkpoint would
# mislabel them. Refuse unless a run dir looks RDP-trained, or the caller says
# explicitly that they mean to run the shared harness on another checkpoint.
# --------------------------------------------------------------------------- #
CKPT_ROOT_EFF="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning/pusht_appraoch2_ckpts"
for ((i=0; i<${#PASSTHRU[@]}; i++)); do
  [[ "${PASSTHRU[$i]}" == "--ckpt_root" ]] && CKPT_ROOT_EFF="${PASSTHRU[$((i+1))]}"
done

if [[ "${ALLOW_NON_RDP}" != "1" ]]; then
  RDP_DIRS=$(find -L "${CKPT_ROOT_EFF}" -mindepth 1 -maxdepth 1 -type d -iname "*rdp*" 2>/dev/null | wc -l)
  if [[ "${RDP_DIRS}" == "0" ]]; then
    echo "[ERROR] no RDP-trained run dir under ${CKPT_ROOT_EFF}" >&2
    echo "        Present there:" >&2
    find -L "${CKPT_ROOT_EFF}" -mindepth 1 -maxdepth 1 -type d -printf "          %f\n" >&2 2>/dev/null
    echo "" >&2
    echo "        RDP is a separate TRAINING run, not an inference mode -- the five" >&2
    echo "        harnesses here are byte-identical, so running one against the uvd or" >&2
    echo "        mix_bspline checkpoint would write non-RDP results into RESULTS_RDP/." >&2
    echo "" >&2
    echo "        Copy the RDP run dir over (on PSC it is" >&2
    echo "        outputs/2026.08.27/11.56.53_push_t_task_APPROACH2_RDP_c1_0.1_206demo" >&2
    echo "        _push_t_task_goal_gmm_aux_rdp), then point --ckpt_root at it." >&2
    echo "        To deliberately run this harness on a non-RDP checkpoint anyway," >&2
    echo "        pass --allow_non_rdp (results are then labelled by run dir, and" >&2
    echo "        are NOT an RDP evaluation)." >&2
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
"${DRIVER}" \
  "${FILTER_ARGS[@]}" \
  --variants    RDP \
  --output_root "${OUTPUT_ROOT}" \
  "${PASSTHRU[@]}"

# Nothing further to do for a plain run or a dry run.
if [[ -z "${REF}" && "${SELF_CHECK}" != "1" ]]; then
  exit 0
fi
for a in "${PASSTHRU[@]}"; do
  [[ "$a" == "--dry_run" ]] && exit 0
done

# --------------------------------------------------------------------------- #
# Locate the results.jsonl just produced
# --------------------------------------------------------------------------- #
find_results() {
  find "$1" -path "*/RDP/*" -name results.jsonl 2>/dev/null | sort
}

mapfile -t PRODUCED < <(find_results "${OUTPUT_ROOT}")
[[ ${#PRODUCED[@]} -gt 0 ]] || fail "no RDP results.jsonl under ${OUTPUT_ROOT}"
if [[ ${#PRODUCED[@]} -gt 1 ]]; then
  echo "[warn] ${#PRODUCED[@]} results.jsonl found; comparing the newest."
  echo "       Narrow with --epochs/--seeds to compare a specific cell."
  mapfile -t PRODUCED < <(printf '%s\n' "${PRODUCED[@]}" | xargs -r ls -1t | head -1)
fi
NEW="${PRODUCED[0]}"

echo
echo "==========================================================================="
echo " Validation"
echo "==========================================================================="
echo " new results : ${NEW}"

# --------------------------------------------------------------------------- #
# --self_check: rerun the SAME cell into a scratch dir and diff against itself.
# Proves the harness is deterministic here before you trust any cross-machine
# comparison. Needs its own output root, since the driver would otherwise skip
# the already-complete cell.
# --------------------------------------------------------------------------- #
if [[ "${SELF_CHECK}" == "1" ]]; then
  SCRATCH="${OUTPUT_ROOT}_SELFCHECK"
  echo " self-check  : rerunning the same config into ${SCRATCH}"
  rm -rf "${SCRATCH}"
  "${DRIVER}" \
    "${FILTER_ARGS[@]}" \
    --variants    RDP \
    --output_root "${SCRATCH}" \
    "${PASSTHRU[@]}"
  mapfile -t SECOND < <(find_results "${SCRATCH}")
  [[ ${#SECOND[@]} -gt 0 ]] || fail "self-check produced no results.jsonl"
  echo
  echo "--- determinism: run 1 vs run 2 ---"
  python3 "${COMPARE}" "${NEW}" "${SECOND[0]}" --tol "${TOL}" --quiet
fi

# --------------------------------------------------------------------------- #
# --compare: diff against the older reference
# --------------------------------------------------------------------------- #
if [[ -n "${REF}" ]]; then
  echo
  echo "--- reference vs new ---"
  python3 "${COMPARE}" "${REF}" "${NEW}" --tol "${TOL}"
fi
