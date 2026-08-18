#!/usr/bin/env bash
# Sync the epoch_99 checkpoint (final checkpoint of each run) for one or more
# PSC training runs into theya_approach2_policies.
#
# How it works:
#   1. Matching run directories are listed directly on psc-data via a remote
#      `ls -d`, which expands the glob on that end (psc-data is a restricted
#      transfer node: ls/rsync/scp only, but that's all this needs -- no
#      python/filesystem-walking finder script required anymore, since we no
#      longer search for a "best" val_loss checkpoint, just the fixed
#      epoch_99.ckpt name).
#   2. For each matched run dir, epoch_99.ckpt is rsync'd from psc-data into
#      theya_approach2_policies/<run_name>/checkpoints/, along with
#      logs.json.txt, the run's .hydra/ dir (resolved config, overrides,
#      hydra.yaml), and any top-level *.yaml configs, for reference.
#
# Usage:
#   ./sync_best_checkpoints.sh '<psc_run_dir_glob>' [local_dest_dir]
#
# Example:
#   ./sync_best_checkpoints.sh \
#     '/jet/home/eswaramo/code/Low_Level_and_Inference/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.08.16/*_hammercleanup_D1_*'
#
# Requires SSH config alias `psc-data` (restricted transfer node: ls/rsync/scp
# only), pointed at the PSC filesystem.

set -euo pipefail

RUN_GLOB="${1:?Usage: $0 '<psc_run_dir_glob>' [local_dest_dir]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DEST="${2:-$REPO_ROOT/theya_approach2_policies}"

DATA_HOST="psc-data"
FINAL_EPOCH=99

echo "==> listing run dirs on $DATA_HOST for pattern: $RUN_GLOB"
RUN_DIRS="$(ssh -o BatchMode=yes "$DATA_HOST" "ls -d $RUN_GLOB" 2>/dev/null || true)"

if [[ -z "$RUN_DIRS" ]]; then
  echo "error: no run dirs matched pattern on $DATA_HOST" >&2
  exit 1
fi

echo "$RUN_DIRS"
echo

mkdir -p "$LOCAL_DEST"

sync_run() {
  local run_dir="$1" run_name ckpt_dir final_ckpt logs_src dest_dir
  run_name="$(basename "$run_dir")"
  ckpt_dir="$run_dir/checkpoints"
  final_ckpt="$ckpt_dir/epoch_${FINAL_EPOCH}.ckpt"
  logs_src="$run_dir/logs.json.txt"
  dest_dir="$LOCAL_DEST/$run_name/checkpoints"

  mkdir -p "$dest_dir"

  echo "==> [$run_name] epoch_${FINAL_EPOCH} checkpoint: $final_ckpt"
  rsync -avP "$DATA_HOST:$final_ckpt" "$dest_dir/" \
    || echo "!! [$run_name] epoch_${FINAL_EPOCH}.ckpt not found/copyable, skipping" >&2

  echo "==> [$run_name] logs.json.txt"
  rsync -avP "$DATA_HOST:$logs_src" "$LOCAL_DEST/$run_name/" \
    || echo "!! [$run_name] logs.json.txt not found, skipping" >&2

  echo "==> [$run_name] .hydra/ (resolved config, overrides, hydra.yaml)"
  rsync -avP "$DATA_HOST:$run_dir/.hydra/" "$LOCAL_DEST/$run_name/.hydra/" \
    || echo "!! [$run_name] .hydra/ not found, skipping" >&2

  echo "==> [$run_name] top-level *.yaml configs"
  rsync -avP --include='*.yaml' --exclude='*' "$DATA_HOST:$run_dir/" "$LOCAL_DEST/$run_name/" \
    || echo "!! [$run_name] no top-level yaml configs found, skipping" >&2

  echo
}

while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue
  sync_run "$run_dir"
done <<< "$RUN_DIRS"

echo "==> done. synced runs into: $LOCAL_DEST"
