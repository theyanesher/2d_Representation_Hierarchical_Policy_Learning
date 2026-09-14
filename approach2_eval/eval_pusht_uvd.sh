#!/usr/bin/env bash
# =============================================================================
# Approach 2 eval -- PushT, UVD run only.
#
#   16.50.09_pusht_APPROACH2_uvd_c1_0.1_206demo_dinov2_DIT_push_t_goal_gmm_aux
#
# Thin wrapper over eval_pusht_approach2_ckpts.sh (which owns the preflight,
# interpreter/overlay wiring, resume and aggregation) so the two run dirs can
# be driven independently without duplicating any of that logic.
#
# Results:  ./RESULTS_UVD/<exp>/<VARIANT>/<epoch>/seed_<seed>/
#
# Any extra flags are forwarded verbatim to the sweep driver, so everything it
# accepts works here too:
#
#   ./eval_pusht_uvd.sh --gpu 1                       # final epoch, 1 seed, 50 eps
#   ./eval_pusht_uvd.sh --gpu 1 --epochs all          # epoch sweep
#   ./eval_pusht_uvd.sh --gpu 1 --seeds 100000,150000,250000
#   ./eval_pusht_uvd.sh --variants all --dry_run      # plan only
#   ./eval_pusht_uvd.sh --gpu none --n_episodes 2     # CPU smoke test
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/eval_pusht_approach2_ckpts.sh"

[[ -x "${DRIVER}" ]] || { echo "[ERROR] driver not executable: ${DRIVER}" >&2; exit 1; }

# 'uvd' is unique to this run dir among the two under the ckpt root. Anything
# the caller passes after these overrides them, because the driver's arg parser
# takes the LAST occurrence of a repeated flag.
exec "${DRIVER}" \
  --exp_filter  "uvd" \
  --output_root "${SCRIPT_DIR}/RESULTS_UVD" \
  "$@"
