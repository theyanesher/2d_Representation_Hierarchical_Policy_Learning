#!/usr/bin/env bash
# =============================================================================
# Approach 2 eval -- PushT, MIX_BSPLINE run only.
#
#   16.50.09_pusht_APPROACH2_mix_bspline_bspline_greville_c1_0.1_206demo
#       _dinov2_DIT_push_t_goal_gmm_aux
#
# Thin wrapper over eval_pusht_approach2_ckpts.sh (which owns the preflight,
# interpreter/overlay wiring, resume and aggregation) so the two run dirs can
# be driven independently without duplicating any of that logic.
#
# Results:  ./RESULTS_MIX_BSPLINE/<exp>/<VARIANT>/<epoch>/seed_<seed>/
#
# Any extra flags are forwarded verbatim to the sweep driver, so everything it
# accepts works here too:
#
#   ./eval_pusht_mix.sh --gpu 1                       # final epoch, 1 seed, 50 eps
#   ./eval_pusht_mix.sh --gpu 1 --epochs all          # epoch sweep
#   ./eval_pusht_mix.sh --gpu 1 --seeds 100000,150000,250000
#   ./eval_pusht_mix.sh --variants all --dry_run      # plan only
#   ./eval_pusht_mix.sh --gpu none --n_episodes 2     # CPU smoke test
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/eval_pusht_approach2_ckpts.sh"

[[ -x "${DRIVER}" ]] || { echo "[ERROR] driver not executable: ${DRIVER}" >&2; exit 1; }

# --exp_filter matches on the run-dir basename; 'mix_bspline' is unique to this
# run. Anything the caller passes after it overrides these defaults, because
# the driver's arg parser takes the LAST occurrence of a repeated flag.
exec "${DRIVER}" \
  --exp_filter  "mix_bspline" \
  --output_root "${SCRIPT_DIR}/RESULTS_MIX_BSPLINE" \
  "$@"
