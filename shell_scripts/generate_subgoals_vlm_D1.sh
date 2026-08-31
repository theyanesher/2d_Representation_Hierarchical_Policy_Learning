#!/usr/bin/env bash
# Generate vlm-only subgoal keypoints for KITCHEN_D1, HAMMER_CLEANUP_D1, and
# COFFEE_PREPERATION_D1 (100 demos each). Deliberately separate from
# generate_subgoals_nonvlm_D1.sh (the other 6 methods + 2 mixes) so this one
# can be started/killed independently -- it depends on the local Qwen server,
# which the other methods don't need at all.
#
# Output tree per task:
#   /data/theya/data/uncertainity_subgoal/D1/EXTRA_KEYPOINTS_vlm_<VLM_TAG>/<TASK>/demo_i/t.npz
# VLM_TAG (default v3) keeps a prompt/parameter revision from resuming into --
# or overwriting -- an earlier one. The original run lives in the untagged
# EXTRA_KEYPOINTS_vlm/, which nothing here touches; set VLM_TAG= (empty) only
# if you deliberately want to write back into it.
#
# WHICH PROMPT THIS RUNS
# subtask_boundaries/prompts.py ships three coarse prompts; the live one
# (COARSE_BOUNDARY_PROMPT) is V3, chosen after an A/B/C run on KITCHEN_D1
# demos 0-4 scored against gripper open/close events:
#
#   V1  5-7 boundaries/ep, stable, but under-segments (misses whole subtasks)
#   V2  12-61 boundaries/ep -- best event recall on the episodes where it
#       behaved, but degenerate (56/61 = every sampled frame) on 2 of 5
#   V3  6-10 boundaries/ep, stable on all 5, and stable on HAMMER_CLEANUP_D1
#       (5-7) with the prompt text byte-identical -- only the instruction
#       string below changes per task
#
# V3 is the default because it is the only arm that is both stable and
# task-agnostic. Its known weakness is timing: boundaries land ~20-30 frames
# early (approach rather than contact), and the dense refinement pass barely
# moves them. Do NOT try to fix that by adding a 'check the gripper fingers in
# every image' rule -- that exact instruction is what makes V2/C2 degenerate;
# see the comment above COARSE_BOUNDARY_PROMPT in subtask_boundaries/prompts.py.
#
# WHY --vlm_camera IS LEFT AT agentview
# A wrist-camera probe (VLM_TAG=v3wrist, KITCHEN_D1 demos 0-2) was meant to make
# finger state legible and fix the early-boundary bias. It made things worse:
# 16/64/16 boundaries, i.e. one fully degenerate episode out of three. The wrist
# view loses the scene context the coarse pass needs. Keep agentview.
#
# WHY THE DEFAULTS BELOW ARE NOT THE GENERATOR'S DEFAULTS
# The first full run (untagged tree) produced 4-7 boundaries per episode and
# sat a median of ~28 frames from the nearest gripper open/close event, with
# only 4-9% of boundaries within 5 frames. Measured causes, and the fix here:
#
#   * No task context ever reached the model: coarse_prompt() shipped a literal
#     '{instruction}' placeholder (fixed in subtask_boundaries/prompts.py), and
#     --vlm_instruction was never passed. Now passed per task, see VLM_INSTRUCTION_*.
#   * --vlm_sample_every_n_frames 15 quantises every boundary to +/-7 frames
#     before refinement even starts. 8 halves that.
#   * --vlm_refinement_radius 15 was SMALLER than the ~28-frame median error, so
#     the dense pass structurally could not reach the true event. 30 covers it.
#   * --vlm_min_boundary_distance_frames 0 let near-duplicate boundaries through;
#     at the higher density the reworked prompt produces, 5 dedupes them.
#
# Stride 8 roughly doubles the images per coarse request (a 705-frame episode
# goes from 3 contact sheets to ~5), so expect this run to take longer per demo
# than the original one did.
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
#
# Env overrides (all optional):
#   VLM_TAG=v4                 write to EXTRA_KEYPOINTS_vlm_v4 instead of _v3
#   EPISODES=5                 short A/B run instead of the full 100 demos
#   SAMPLE_EVERY=15            coarse sampling stride
#   REFINE_RADIUS=15           dense refinement half-window
#   MIN_BOUNDARY_DIST=0        drop boundaries closer together than this
#   EXTRA_ARGS="--force"       anything else forwarded to the generator

set -euo pipefail

REPO_ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
ENV_PY="${REPO_ROOT}/.pixi/envs/eval/bin/python"
DATA_ROOT="/data/theya/data/uncertainity_subgoal/D1"

VLM_TAG="${VLM_TAG-v3}"
EPISODES="${EPISODES:-100}"
SAMPLE_EVERY="${SAMPLE_EVERY:-8}"
REFINE_RADIUS="${REFINE_RADIUS:-30}"
MIN_BOUNDARY_DIST="${MIN_BOUNDARY_DIST:-5}"
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

# Per-task instruction, taken from the mimicgen env classes themselves
# (CoffeePreparation / HammerCleanup_D0 docstrings, Kitchen_D0._check_success)
# rather than guessed, so the model is told the actual task it is watching.
VLM_INSTRUCTION_KITCHEN_D1="A robot arm prepares food in a kitchen: it presses the stove button to turn the stove on, picks up the pot and places it on the stove burner, picks up the bread and puts it in the pot, then moves the pot to the serving region and turns the stove off."
VLM_INSTRUCTION_HAMMER_CLEANUP_D1="A robot arm cleans up a hammer: it opens the drawer, picks up the hammer from the table, places the hammer inside the drawer, and closes the drawer."
VLM_INSTRUCTION_COFFEE_PREPERATION_D1="A robot arm prepares coffee: it opens the drawer, takes out the coffee pod and the mug, opens the coffee machine lid, places the pod in the machine and the mug on the machine base, then closes the coffee machine lid."

instruction_for_task() {
  case "$1" in
    KITCHEN_D1)              printf '%s' "${VLM_INSTRUCTION_KITCHEN_D1}" ;;
    HAMMER_CLEANUP_D1)       printf '%s' "${VLM_INSTRUCTION_HAMMER_CLEANUP_D1}" ;;
    COFFEE_PREPERATION_D1)   printf '%s' "${VLM_INSTRUCTION_COFFEE_PREPERATION_D1}" ;;
    *)                       printf '' ;;
  esac
}

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

SUFFIX_ARGS=()
if [[ -n "${VLM_TAG}" ]]; then
  SUFFIX_ARGS=(--dirname_suffix "${VLM_TAG}")
fi

echo "[generate_subgoals_vlm] tasks=${TASKS[*]}"
echo "[generate_subgoals_vlm] tag=${VLM_TAG:-<none>} episodes=${EPISODES} stride=${SAMPLE_EVERY}" \
     "refine_radius=${REFINE_RADIUS} min_boundary_dist=${MIN_BOUNDARY_DIST}"

for TASK in "${TASKS[@]}"; do
  TASK_DIR="${DATA_ROOT}/${TASK}"
  if [[ ! -d "${TASK_DIR}" ]]; then
    echo "[ERROR] task dir not found: ${TASK_DIR}" >&2
    exit 1
  fi

  INSTRUCTION="$(instruction_for_task "${TASK}")"
  if [[ -z "${INSTRUCTION}" ]]; then
    # Better to stop than to silently fall back to the generic no-instruction
    # wording, which is exactly the context-free setup that under-segmented.
    echo "[ERROR] no VLM instruction defined for task ${TASK} -- add one to this script" >&2
    exit 1
  fi

  echo ""
  echo "=================================================================="
  echo "[generate_subgoals_vlm] task=${TASK}"
  echo "[generate_subgoals_vlm] instruction: ${INSTRUCTION}"
  echo "=================================================================="

  "${ENV_PY}" external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py \
    --data_root "${DATA_ROOT}" \
    --task "${TASK}" \
    --episodes "${EPISODES}" \
    --methods vlm \
    --vlm_provider qwen \
    --vlm_instruction "${INSTRUCTION}" \
    --vlm_sample_every_n_frames "${SAMPLE_EVERY}" \
    --vlm_refinement_radius "${REFINE_RADIUS}" \
    --vlm_min_boundary_distance_frames "${MIN_BOUNDARY_DIST}" \
    "${SUFFIX_ARGS[@]}" \
    ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"} \
    --dump_indices
done

echo ""
echo "[generate_subgoals_vlm] done: ${TASKS[*]}"
