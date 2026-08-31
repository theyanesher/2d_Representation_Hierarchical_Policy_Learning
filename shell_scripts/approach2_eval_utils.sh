#!/usr/bin/env bash
# Shared output layout for Approach 2 evaluation launchers.
# Source this file after LL_EXP_DIR, checkpoint, and OUTPUT_BASE are defined.

approach2_prepare_eval_layout() {
  if (($# != 4)); then
    echo "approach2_prepare_eval_layout: expected POLICY_DIR CHECKPOINT SCRIPT EVAL_NAME" >&2
    return 2
  fi

  local policy_dir="$1"
  local checkpoint="$2"
  local source_script="$3"
  local eval_name="$4"
  local checkpoint_tag script_copy

  checkpoint_tag="${checkpoint##*/}"
  checkpoint_tag="${checkpoint_tag%.ckpt}"
  [[ -n "$checkpoint_tag" ]] || checkpoint_tag="unknown_checkpoint"

  # OUTPUT_BASE values are already filesystem-friendly in the current scripts,
  # but sanitize here so new launchers cannot accidentally create subtrees.
  eval_name="${eval_name//[^A-Za-z0-9._-]/_}"
  [[ -n "$eval_name" ]] || eval_name="$(basename "${source_script%.sh}")"

  APPROACH2_POLICY_DIR="$policy_dir"
  APPROACH2_CHECKPOINT_TAG="$checkpoint_tag"
  APPROACH2_EVAL_ROOT="${policy_dir}/evaluations/${checkpoint_tag}/${eval_name}"
  APPROACH2_EVAL_LOG="${APPROACH2_EVAL_ROOT}/eval.log"
  APPROACH2_SUMMARY_JSON="${APPROACH2_EVAL_ROOT}/summary_all_seeds.json"
  APPROACH2_SUMMARY_TEXT="${APPROACH2_EVAL_ROOT}/summary_all_seeds.txt"

  mkdir -p "$APPROACH2_EVAL_ROOT"
  script_copy="${APPROACH2_EVAL_ROOT}/$(basename "$source_script")"
  if [[ ! -e "$script_copy" || ! "$source_script" -ef "$script_copy" ]]; then
    cp -p "$source_script" "$script_copy"
  fi

  {
    printf 'policy_dir=%s\n' "$policy_dir"
    printf 'checkpoint=%s\n' "$checkpoint"
    printf 'source_script=%s\n' "$source_script"
    printf 'archived_script=%s\n' "$script_copy"
  } > "${APPROACH2_EVAL_ROOT}/eval_manifest.txt"

  # Existing launchers build outputs relative to SCRIPT_DIR. Redirecting this
  # one variable preserves their resume logic while moving every artifact.
  SCRIPT_DIR="$APPROACH2_EVAL_ROOT"

  export APPROACH2_POLICY_DIR APPROACH2_CHECKPOINT_TAG APPROACH2_EVAL_ROOT
  export APPROACH2_EVAL_LOG APPROACH2_SUMMARY_JSON APPROACH2_SUMMARY_TEXT
}

approach2_start_eval_logging() {
  [[ -n "${APPROACH2_EVAL_LOG:-}" ]] \
    || { echo "approach2_start_eval_logging: layout is not initialized" >&2; return 2; }

  # Append so reruns and resumed evaluations retain the complete history.
  exec > >(tee -a "$APPROACH2_EVAL_LOG") 2>&1
  echo "[eval-layout] policy:     $APPROACH2_POLICY_DIR"
  echo "[eval-layout] checkpoint: $APPROACH2_CHECKPOINT_TAG"
  echo "[eval-layout] outputs:    $APPROACH2_EVAL_ROOT"
  echo "[eval-layout] log:        $APPROACH2_EVAL_LOG"
}

approach2_write_combined_summary() {
  [[ -n "${APPROACH2_EVAL_ROOT:-}" ]] \
    || { echo "approach2_write_combined_summary: layout is not initialized" >&2; return 2; }

  local python_bin="${ENV_PY:-python3}"
  local summary_tool="${INFERENCE_ROOT}/scripts/summarize_approach2_evals.py"
  local -a expected_seeds=("$@")
  if ((${#expected_seeds[@]} == 0)); then
    expected_seeds=(100000 150000 250000)
  fi
  "$python_bin" "$summary_tool" \
    --eval-dir "$APPROACH2_EVAL_ROOT" \
    --checkpoint "$APPROACH2_CHECKPOINT_TAG" \
    --expected-seeds "${expected_seeds[@]}"
}
