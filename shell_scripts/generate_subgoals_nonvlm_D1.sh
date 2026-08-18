#!/usr/bin/env bash
# Generate subgoal keypoints for KITCHEN_D1, HAMMER_CLEANUP_D1, and
# COFFEE_PREPERATION_D1 (100 demos each) -- everything EXCEPT vlm, which is
# deliberately split into generate_subgoals_vlm_D1.sh so it can be run/killed
# independently of this (it needs the separate local Qwen server; this script
# needs none of that and has no GPU-sharing concerns with it).
#
# One generate_extra_keypoints.py run per task, producing:
#   awe                                    (--awe_solver dp -- see note below)
#   bspline, bspline_greville               + mix_bspline_bspline_greville      (window 10)
#   gripper_heuristic, orientation_heuristic + mix_gripper_heuristic_orientation_heuristic (window 10)
#   uvd            (--uvd_preprocessor vip)
#
# Output tree per task (verified with a real --episodes 0 dry run of this
# exact flag set, so this is the literal name, not a guess):
#   /data/theya/data/uncertainity_subgoal/D1/
#     EXTRA_KEYPOINTS_bspline_bspline_greville_awe_gripper_heuristic_orientation_heuristic_uvd_mix_bspline_bspline_greville_mix_gripper_heuristic_orientation_heuristic/
#       <TASK>/demo_i/t.npz
# This is a DIFFERENT (and separate) mirror tree from vlm's own
# EXTRA_KEYPOINTS_vlm/<TASK>/ (see generate_subgoals_vlm_D1.sh) -- decoupling
# the two runs means they no longer share one combined tree; each npz still
# carries the same underlying original data, just under two directories
# instead of one.
#
# --mix_window applies to BOTH --mix_methods groups (it's a single shared
# value, not per-group) -- 10 satisfies "window 10" for both as requested.
#
# awe_solver=dp: optimal but O(T^3) -- the docs warn it's for "short demos
# only, roughly <= a few hundred frames" and these tasks' demos run
# ~580-650 frames, so this will be considerably slower than --awe_solver
# greedy across 100 demos x 3 tasks. Left as dp deliberately; switch to
# greedy here if runtime becomes a problem.
#
# Usage:
#   ./shell_scripts/generate_subgoals_nonvlm_D1.sh [TASK ...]
# With no args, runs all three tasks. Pass one or more task names to run a
# subset, e.g. only KITCHEN_D1.

set -euo pipefail

REPO_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
ENV_PY="${REPO_ROOT}/.pixi/envs/eval/bin/python"
DATA_ROOT="/data/theya/data/uncertainity_subgoal/D1"

DEFAULT_TASKS=(KITCHEN_D1 HAMMER_CLEANUP_D1 COFFEE_PREPERATION_D1)
TASKS=("${@:-${DEFAULT_TASKS[@]}}")

if [[ ! -e "${ENV_PY}" ]]; then
  echo "[ERROR] missing python: ${ENV_PY}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${DATA_ROOT}/${TASK}"
  if [[ ! -d "${TASK_DIR}" ]]; then
    echo "[ERROR] task dir not found: ${TASK_DIR}" >&2
    exit 1
  fi

  echo ""
  echo "=================================================================="
  echo "[generate_subgoals_nonvlm] task=${TASK}"
  echo "=================================================================="

  "${ENV_PY}" external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root "${DATA_ROOT}" \
    --task "${TASK}" \
    --episodes 100 \
    --methods awe bspline bspline_greville gripper_heuristic orientation_heuristic uvd \
    --mix_methods bspline bspline_greville \
    --mix_methods gripper_heuristic orientation_heuristic \
    --mix_window 10 \
    --awe_solver greedy \
    --uvd_preprocessor vip \
    --uvd_device cuda:0 \
    --dump_indices
done

echo ""
echo "[generate_subgoals_nonvlm] done: ${TASKS[*]}"
