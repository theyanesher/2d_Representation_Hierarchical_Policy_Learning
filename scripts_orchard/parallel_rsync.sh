#!/usr/bin/env bash
set -euo pipefail

#################### CONFIG ####################
REMOTE_HOST="ywang59@bridges2.psc.edu"
REMOTE_BASE="/jet/projects/cis240052p/ywang59/multitask_all_training_data/"
LOCAL_BASE="/tmp/pick_and_place/inside_link_cgn_grasp_0101_grasp_only"

MAX_JOBS=20

# Exclude local folder name prefixes (top-level only)
EXCLUDE_RE='^(0822-|0815-|0826-|1026-|1119-)'
###############################################

# Ensure remote base exists
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE'"

# List local top-level folders (names only), apply exclude
mapfile -t FOLDERS < <(
  find "$LOCAL_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | grep -Ev "$EXCLUDE_RE" || true
)

echo "Found ${#FOLDERS[@]} local folders to sync."

rsync_one() {
  local folder="$1"
  echo "[START] $folder"

  # Ensure remote subdir exists (optional; rsync can also create it if parent exists)
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/$folder'"

  rsync -azP --partial --inplace \
    "$LOCAL_BASE/$folder/" \
    "$REMOTE_HOST:$REMOTE_BASE/$folder/"

  echo "[DONE] $folder"
}

export -f rsync_one
export REMOTE_HOST REMOTE_BASE LOCAL_BASE

printf "%s\n" "${FOLDERS[@]}" \
  | xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'rsync_one "$@"' _ {}

echo "All done."
