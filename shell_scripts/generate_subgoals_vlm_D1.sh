#!/usr/bin/env bash
# Generate vlm-only subgoal keypoints for KITCHEN_D1, HAMMER_CLEANUP_D1, and
# COFFEE_PREPERATION_D1 (100 demos each). Deliberately separate from
# generate_subgoals_nonvlm_D1.sh (the other 6 methods + 2 mixes) so this one
# can be started/killed independently -- it depends on the local Qwen server,
# which the other methods don't need at all.
#
# Output tree per task:
#   /data/theya/data/uncertainity_subgoal/D1/EXTRA_KEYPOINTS_vlm/<TASK>/demo_i/t.npz
# A separate mirror tree from generate_subgoals_nonvlm_D1.sh's combined one
# (see that script's header) -- this is the tradeoff for decoupling the two
# runs: two trees instead of one, each npz still carrying the same
# underlying original data.
#
# You must start the local Qwen server yourself BEFORE running this --
# nothing here starts it automatically:
#   ./scripts/serve_qwen_local.sh   (one-time setup: ./scripts/setup_qwen_local.sh)
# Stop it independently whenever you want with:
#   pkill -f 'vllm serve'
# Uses the default endpoint http://127.0.0.1:8000/v1, so no --vlm_qwen_base_url
# override needed here.
#
# Usage:
#   ./shell_scripts/generate_subgoals_vlm_D1.sh [TASK ...]
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

if ! curl -sf -m 3 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] local Qwen server not reachable at http://127.0.0.1:8000/v1" >&2
  echo "        start it first: ./scripts/serve_qwen_local.sh" >&2
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
  echo "[generate_subgoals_vlm] task=${TASK}"
  echo "=================================================================="

  "${ENV_PY}" external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root "${DATA_ROOT}" \
    --task "${TASK}" \
    --episodes 100 \
    --methods vlm \
    --vlm_provider qwen \
    --dump_indices
done

echo ""
echo "[generate_subgoals_vlm] done: ${TASKS[*]}"
