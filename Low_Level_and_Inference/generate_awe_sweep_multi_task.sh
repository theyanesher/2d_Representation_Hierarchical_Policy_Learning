#!/usr/bin/env bash
# Sweep AWE subgoal keypoints over four (solver, err_threshold) settings for
# KITCHEN_D1, HAMMER_CLEANUP_D1 and COFFEE_PREPERATION_D1.
#
#   greedy  0.5      fast
#   greedy  0.2      fast
#   dp      0.35     slow -- O(T^3), see note below
#   dp      0.05     slow
#
# Each setting lands in its own mirror tree, because generate_extra_keypoints.py
# now spells awe as awe-<solver>-th<threshold> in the directory name:
#
#   <DATA_ROOT>/EXTRA_KEYPOINTS_awe-greedy-th0.5/<TASK>/demo_i/t.npz
#   <DATA_ROOT>/EXTRA_KEYPOINTS_awe-greedy-th0.2/<TASK>/...
#   <DATA_ROOT>/EXTRA_KEYPOINTS_awe-dp-th0.35/<TASK>/...
#   <DATA_ROOT>/EXTRA_KEYPOINTS_awe-dp-th0.05/<TASK>/...
#
# So the four settings never collide with each other, and none of them touches
# the existing EXTRA_KEYPOINTS_awe/ or the big combined tree. The npz key stays
# goal_gripper_pcd_awe in every tree, so downstream configs read them by
# switching --goal_root, not by renaming keys.
#
# The greedy settings run first so their results are on disk long before the
# dp ones finish. AWE dp is O(T^3) and these demos are 296-705 frames, so dp
# costs minutes per demo where greedy costs a fraction of a second; the two dp
# settings dominate the total wall time. NUM_WORKERS parallelises ACROSS demos
# (one process per demo), which is the only axis available -- a single demo's
# dp search is inherently sequential.
#
# Usage:
#   ./Low_Level_and_Inference/generate_awe_sweep_multi_task.sh                # all 4 settings, all 3 tasks
#   ./Low_Level_and_Inference/generate_awe_sweep_multi_task.sh KITCHEN_D1     # subset of tasks
#   SETTINGS="greedy:0.5 greedy:0.2" ./..._sweep_multi_task.sh                # subset of settings
#   NUM_WORKERS=32 ./..._sweep_multi_task.sh                                  # more demo workers
#
# Resumable: generate_extra_keypoints.py skips demos already complete in the
# target tree, so re-running after a kill picks up where it left off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/theya/data/uncertainity_subgoal/D1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPISODES="${EPISODES:-100}"
MULTI_TASK="${SCRIPT_DIR}/generate_extra_keypoints_multi_task.sh"

DEFAULT_TASKS=(KITCHEN_D1 HAMMER_CLEANUP_D1 COFFEE_PREPERATION_D1)
TASKS=("${@:-${DEFAULT_TASKS[@]}}")

# solver:threshold, greedy first so the cheap results land early.
read -r -a SETTINGS_ARR <<< "${SETTINGS:-greedy:0.5 greedy:0.2 dp:0.35 dp:0.05}"

if [[ ! -x "${MULTI_TASK}" ]]; then
  echo "[ERROR] multi-task wrapper not executable: ${MULTI_TASK}" >&2
  exit 1
fi

# Validate everything up front -- a typo must not surface hours into the dp runs.
for TASK in "${TASKS[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${TASK}" ]]; then
    echo "[ERROR] task directory not found: ${DATA_ROOT}/${TASK}" >&2
    exit 1
  fi
done
for SETTING in "${SETTINGS_ARR[@]}"; do
  SOLVER="${SETTING%%:*}"
  THRESH="${SETTING##*:}"
  if [[ "${SOLVER}" != "greedy" && "${SOLVER}" != "dp" ]]; then
    echo "[ERROR] bad solver in setting '${SETTING}' (want greedy or dp)" >&2
    exit 1
  fi
  if [[ ! "${THRESH}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "[ERROR] bad threshold in setting '${SETTING}'" >&2
    exit 1
  fi
done

echo "[awe-sweep] tasks:    ${TASKS[*]}"
echo "[awe-sweep] settings: ${SETTINGS_ARR[*]}"
echo "[awe-sweep] workers:  ${NUM_WORKERS} | episodes: ${EPISODES} | data_root: ${DATA_ROOT}"

for SETTING in "${SETTINGS_ARR[@]}"; do
  SOLVER="${SETTING%%:*}"
  THRESH="${SETTING##*:}"
  echo ""
  echo "=================================================================="
  echo "[awe-sweep] solver=${SOLVER} err_threshold=${THRESH}"
  echo "=================================================================="
  START=$(date +%s)
  DATA_ROOT="${DATA_ROOT}" "${MULTI_TASK}" "${TASKS[@]}" -- \
    --methods awe \
    --awe_solver "${SOLVER}" \
    --awe_err_threshold "${THRESH}" \
    --episodes "${EPISODES}" \
    --num_workers "${NUM_WORKERS}" \
    --dump_indices
  echo "[awe-sweep] solver=${SOLVER} th=${THRESH} took $(( $(date +%s) - START ))s"
done

echo ""
echo "[awe-sweep] all settings completed: ${SETTINGS_ARR[*]}"
