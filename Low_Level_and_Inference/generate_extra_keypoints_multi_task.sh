#!/usr/bin/env bash
# Run generate_extra_keypoints.py sequentially for multiple task directories.
#
# Usage:
#   ./generate_extra_keypoints_multi_task.sh TASK_1 TASK_2 [TASK_3 ...]
#
# Pass custom generator options after `--`:
#   ./generate_extra_keypoints_multi_task.sh TASK_1 TASK_2 -- \
#     --methods rdp awe \
#     --awe_solver greedy \
#     --num_workers 8 \
#     --dump_indices
#
# Without custom options, the wrapper generates the two heuristic methods
# using eight demo workers. Override the dataset or Python executable with:
#   DATA_ROOT=/path/to/data PYTHON_BIN=/path/to/python ./generate_extra_keypoints_multi_task.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data/theya/data/uncertainity_subgoal/D1}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.pixi/envs/eval/bin/python}"
GENERATOR="${REPO_ROOT}/external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py"

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

if (( $# == 0 )) || [[ "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

TASKS=()
GENERATOR_ARGS=()
READING_GENERATOR_ARGS=0

while (( $# > 0 )); do
  if [[ "${1}" == "--" && "${READING_GENERATOR_ARGS}" == 0 ]]; then
    READING_GENERATOR_ARGS=1
  elif (( READING_GENERATOR_ARGS )); then
    GENERATOR_ARGS+=("${1}")
  else
    TASKS+=("${1}")
  fi
  shift
done

if (( ${#TASKS[@]} == 0 )); then
  echo "[ERROR] provide at least one task before --" >&2
  usage >&2
  exit 2
fi

if (( ${#GENERATOR_ARGS[@]} == 0 )); then
  GENERATOR_ARGS=(
    --methods gripper_heuristic orientation_heuristic
    --num_workers 8
    --dump_indices
  )
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
  echo "[ERROR] generator script not found: ${GENERATOR}" >&2
  exit 1
fi

# Validate every task before starting, so a typo cannot cause a partial run.
for TASK in "${TASKS[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${TASK}" ]]; then
    echo "[ERROR] task directory not found: ${DATA_ROOT}/${TASK}" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"

for TASK in "${TASKS[@]}"; do
  echo "[multi-task] starting ${TASK}"
  "${PYTHON_BIN}" "${GENERATOR}" \
    --data_root "${DATA_ROOT}" \
    --task "${TASK}" \
    "${GENERATOR_ARGS[@]}"
  echo "[multi-task] completed ${TASK}"
done

echo "[multi-task] all tasks completed: ${TASKS[*]}"
