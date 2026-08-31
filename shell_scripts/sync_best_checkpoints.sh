#!/usr/bin/env bash
# Copy a sampled set of checkpoints from PSC via the psc-data transfer node.
#
# Each remote run is mapped automatically to:
#   theya_approach2_policies/<remote-run-directory-name>/
#
# Checkpoints whose numeric epoch satisfies `epoch % K == OFFSET` are copied.
# The newest available checkpoint is also copied by default, which keeps runs
# such as epoch_0 ... epoch_99 useful when K does not divide 99.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_HOST="psc-data"
LOCAL_DEST="$REPO_ROOT/theya_approach2_policies"
EVERY=10
OFFSET=0
JOBS=2
INCLUDE_LAST=1
COPY_METADATA=1
COPY_EVAL_SCRIPTS=1
DRY_RUN=0
FROM_FILE=""

usage() {
  cat <<'EOF'
Usage:
  sync_best_checkpoints.sh [options] RUN_DIR_OR_GLOB [RUN_DIR_OR_GLOB ...]
  sync_best_checkpoints.sh [options] --from-file RUN_LIST.txt

Copy epoch checkpoints from one or more PSC run directories through the
`psc-data` SSH alias. Quote globs so they expand on PSC, not locally.

Options:
  -k, --every K       Copy epochs N where N % K == OFFSET (default: 10)
      --offset N      Sampling offset in [0, K) (default: 0)
      --include-last  Also copy the largest epoch in each run (default)
      --no-last       Do not add the largest epoch automatically
  -j, --jobs N        Runs to transfer concurrently (default: 2)
  -d, --dest DIR      Local policy root (default: ./theya_approach2_policies)
  -H, --host HOST     Transfer-node SSH alias (default: psc-data)
  -f, --from-file F   Read run directories/globs from F, one per line
      --no-metadata   Skip logs.json.txt, .hydra/, and top-level YAML files
      --no-eval-scripts
                      Do not archive matching Approach 2 eval launchers
  -n, --dry-run       Resolve runs and print the exact mapping; copy nothing
  -h, --help          Show this help

Examples:
  # One checkpoint every 10 epochs, plus the final checkpoint:
  ./shell_scripts/sync_best_checkpoints.sh -k 10 \
    '/jet/home/eswaramo/code/project/outputs/2026.08.24/*'

  # Several explicit runs:
  ./shell_scripts/sync_best_checkpoints.sh -k 5 /remote/run_A /remote/run_B

  # A newline-delimited list (blank lines and # comments are ignored):
  ./shell_scripts/sync_best_checkpoints.sh -k 20 --from-file runs.txt

The operation is resumable and non-destructive: files are copied, never
removed from PSC. Existing complete files are skipped by rsync.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

declare -a INPUT_PATTERNS=()

while (($#)); do
  case "$1" in
    -k|--every)
      (($# >= 2)) || die "$1 requires a value"
      EVERY="$2"
      shift 2
      ;;
    --offset)
      (($# >= 2)) || die "$1 requires a value"
      OFFSET="$2"
      shift 2
      ;;
    --include-last)
      INCLUDE_LAST=1
      shift
      ;;
    --no-last)
      INCLUDE_LAST=0
      shift
      ;;
    -j|--jobs)
      (($# >= 2)) || die "$1 requires a value"
      JOBS="$2"
      shift 2
      ;;
    -d|--dest)
      (($# >= 2)) || die "$1 requires a value"
      LOCAL_DEST="$2"
      shift 2
      ;;
    -H|--host)
      (($# >= 2)) || die "$1 requires a value"
      DATA_HOST="$2"
      shift 2
      ;;
    -f|--from-file)
      (($# >= 2)) || die "$1 requires a value"
      FROM_FILE="$2"
      shift 2
      ;;
    --no-metadata)
      COPY_METADATA=0
      shift
      ;;
    --no-eval-scripts)
      COPY_EVAL_SCRIPTS=0
      shift
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      INPUT_PATTERNS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      INPUT_PATTERNS+=("$1")
      shift
      ;;
  esac
done

is_positive_integer "$EVERY" || die "--every must be a positive integer"
is_nonnegative_integer "$OFFSET" || die "--offset must be a non-negative integer"
((OFFSET < EVERY)) || die "--offset must be smaller than --every"
is_positive_integer "$JOBS" || die "--jobs must be a positive integer"
[[ "$DATA_HOST" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] \
  || die "invalid SSH host/alias: $DATA_HOST"

if [[ -n "$FROM_FILE" ]]; then
  [[ -r "$FROM_FILE" ]] || die "cannot read run list: $FROM_FILE"
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim surrounding whitespace; paths themselves intentionally cannot
    # contain whitespace because psc-data expands these as remote globs.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    INPUT_PATTERNS+=("$line")
  done < "$FROM_FILE"
fi

((${#INPUT_PATTERNS[@]} > 0)) || die "provide at least one run directory/glob"

# These patterns are interpolated into a deliberately small remote `ls`
# command so that '*' expands on PSC. Reject shell syntax and whitespace.
validate_remote_pattern() {
  local pattern="$1"
  local allowed='^[][A-Za-z0-9_./*?+@%=,-]+$'
  [[ "$pattern" == /* ]] || die "PSC paths must be absolute: $pattern"
  [[ "$pattern" =~ $allowed ]] \
    || die "unsupported character in PSC path/glob: $pattern"
}

for pattern in "${INPUT_PATTERNS[@]}"; do
  validate_remote_pattern "$pattern"
done

command -v ssh >/dev/null 2>&1 || die "ssh is not installed"
command -v rsync >/dev/null 2>&1 || die "rsync is not installed"

SSH_OPTIONS=(-o BatchMode=yes -o Compression=no)

echo "==> Resolving ${#INPUT_PATTERNS[@]} run input(s) on $DATA_HOST"
remote_ls_command="ls -1d"
for pattern in "${INPUT_PATTERNS[@]}"; do
  remote_ls_command+=" $pattern"
done

# `ls` can return nonzero when one glob misses while still printing matches
# for the other inputs, so decide based on the collected output instead.
run_output="$(ssh "${SSH_OPTIONS[@]}" "$DATA_HOST" "$remote_ls_command" 2>/dev/null || true)"

declare -a RUN_DIRS=()
while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue
  validate_remote_pattern "$run_dir"
  RUN_DIRS+=("${run_dir%/}")
done <<< "$run_output"

if ((${#RUN_DIRS[@]} == 0)); then
  die "no remote run directories matched"
fi

# Sort and remove duplicates without relying on the remote host's utilities.
mapfile -t RUN_DIRS < <(printf '%s\n' "${RUN_DIRS[@]}" | LC_ALL=C sort -u)
echo "==> Found ${#RUN_DIRS[@]} unique run(s)"

# List every checkpoint for every resolved run in one SSH round trip.
checkpoint_ls_command="ls -1d"
for run_dir in "${RUN_DIRS[@]}"; do
  checkpoint_ls_command+=" $run_dir/checkpoints/epoch_*.ckpt"
done
checkpoint_output="$(ssh "${SSH_OPTIONS[@]}" "$DATA_HOST" "$checkpoint_ls_command" 2>/dev/null || true)"

declare -A CHECKPOINTS_BY_RUN=()
while IFS= read -r checkpoint; do
  [[ -z "$checkpoint" ]] && continue
  filename="${checkpoint##*/}"
  [[ "$filename" =~ ^epoch_([0-9]+)\.ckpt$ ]] || continue
  run_dir="${checkpoint%/checkpoints/$filename}"
  [[ -n "${CHECKPOINTS_BY_RUN[$run_dir]:-}" ]] \
    && CHECKPOINTS_BY_RUN[$run_dir]+=$'\n'
  CHECKPOINTS_BY_RUN[$run_dir]+="$checkpoint"
done <<< "$checkpoint_output"

# Allocate readable local names. A basename is used normally; if two inputs
# have the same basename, prepend parent directory names until it is unique.
declare -A USED_NAMES=()
declare -a ACTIVE_RUNS=()
declare -a DEST_NAMES=()
declare -a SELECTED_NAMES=()

allocate_dest_name() {
  local source="$1" base parent candidate prefix
  base="${source##*/}"
  candidate="$base"
  parent="${source%/*}"
  prefix=""

  while [[ -n "${USED_NAMES[$candidate]:-}" ]]; do
    [[ "$parent" != "/" && -n "$parent" ]] || break
    if [[ -z "$prefix" ]]; then
      prefix="${parent##*/}"
    else
      prefix="${parent##*/}__$prefix"
    fi
    candidate="${prefix}__${base}"
    parent="${parent%/*}"
  done

  if [[ -n "${USED_NAMES[$candidate]:-}" ]]; then
    # This should only be reachable for duplicate identical inputs (normally
    # removed above), but retain a deterministic, collision-safe fallback.
    candidate="${base}__$(printf '%s' "$source" | cksum | awk '{print $1}')"
  fi

  USED_NAMES[$candidate]="$source"
  ALLOCATED_DEST_NAME="$candidate"
}

for run_dir in "${RUN_DIRS[@]}"; do
  checkpoint_list="${CHECKPOINTS_BY_RUN[$run_dir]:-}"
  if [[ -z "$checkpoint_list" ]]; then
    echo "warning: no epoch_N.ckpt files found; skipping $run_dir" >&2
    continue
  fi

  declare -a selected=()
  max_epoch=-1
  max_name=""
  while IFS= read -r checkpoint; do
    filename="${checkpoint##*/}"
    [[ "$filename" =~ ^epoch_([0-9]+)\.ckpt$ ]] || continue
    epoch=$((10#${BASH_REMATCH[1]}))
    if ((epoch > max_epoch)); then
      max_epoch=$epoch
      max_name="$filename"
    fi
    if (((epoch - OFFSET) >= 0 && (epoch - OFFSET) % EVERY == 0)); then
      selected+=("$filename")
    fi
  done <<< "$checkpoint_list"

  if ((INCLUDE_LAST)) && [[ -n "$max_name" ]]; then
    already_selected=0
    for filename in "${selected[@]}"; do
      [[ "$filename" == "$max_name" ]] && already_selected=1
    done
    ((already_selected)) || selected+=("$max_name")
  fi

  if ((${#selected[@]} == 0)); then
    echo "warning: no checkpoints matched sampling rule; skipping $run_dir" >&2
    continue
  fi

  mapfile -t selected < <(
    printf '%s\n' "${selected[@]}" \
      | sort -t_ -k2,2n -u
  )
  allocate_dest_name "$run_dir"
  dest_name="$ALLOCATED_DEST_NAME"
  ACTIVE_RUNS+=("$run_dir")
  DEST_NAMES+=("$dest_name")
  SELECTED_NAMES+=("$(printf '%s\n' "${selected[@]}")")
done

((${#ACTIVE_RUNS[@]} > 0)) || die "none of the matched runs had selectable checkpoints"

echo
echo "Transfer plan (epoch % $EVERY == $OFFSET; include-last=$INCLUDE_LAST):"
for ((i = 0; i < ${#ACTIVE_RUNS[@]}; i++)); do
  checkpoint_csv="${SELECTED_NAMES[$i]//$'\n'/, }"
  echo "  ${ACTIVE_RUNS[$i]}"
  echo "    -> $LOCAL_DEST/${DEST_NAMES[$i]}"
  echo "       $checkpoint_csv"
done

if ((DRY_RUN)); then
  echo
  echo "==> Dry run complete; no local files were changed"
  exit 0
fi

mkdir -p "$LOCAL_DEST"

stage_eval_scripts() {
  local run_dir="$1" dest_dir="$2" run_name script relative target count=0
  local -a matches=()
  run_name="${run_dir##*/}"

  # Newer launchers declare LL_RUN_NAME; the two multi-checkpoint launchers
  # embed the same run name directly in LL_EXP_DIR. A fixed-string scan covers
  # both without executing or trying to parse the eval scripts.
  if command -v rg >/dev/null 2>&1; then
    mapfile -t matches < <(
      rg -l -F "$run_name" "$REPO_ROOT/shell_scripts" \
        --glob '*.sh' --glob '!sync_best_checkpoints.sh' || true
    )
  else
    mapfile -t matches < <(
      grep -rlF --include='*.sh' "$run_name" "$REPO_ROOT/shell_scripts" || true
    )
  fi

  for script in "${matches[@]}"; do
    [[ -f "$script" ]] || continue
    relative="${script#"$REPO_ROOT/shell_scripts/"}"
    target="$dest_dir/eval_scripts/$relative"
    mkdir -p "$(dirname "$target")"
    cp -p "$script" "$target"
    ((count += 1))
  done

  if ((count)); then
    mkdir -p "$dest_dir/eval_scripts/_shared"
    cp -p "$REPO_ROOT/shell_scripts/approach2_eval_utils.sh" \
      "$dest_dir/eval_scripts/_shared/"
    cp -p "$REPO_ROOT/scripts/summarize_approach2_evals.py" \
      "$dest_dir/eval_scripts/_shared/"
    echo "==> [${run_name}] archived $count matching eval script(s)"
  else
    echo "==> [${run_name}] no matching eval scripts found locally"
  fi
}

sync_run() {
  local index="$1" run_dir dest_name dest_dir selected_text filename
  local -a filters rsync_args
  run_dir="${ACTIVE_RUNS[$index]}"
  dest_name="${DEST_NAMES[$index]}"
  selected_text="${SELECTED_NAMES[$index]}"
  dest_dir="$LOCAL_DEST/$dest_name"
  mkdir -p "$dest_dir"

  filters=(--include='/checkpoints/')
  while IFS= read -r filename; do
    [[ -n "$filename" ]] || continue
    filters+=(--include="/checkpoints/$filename")
  done <<< "$selected_text"
  filters+=(--exclude='/checkpoints/*')

  if ((COPY_METADATA)); then
    filters+=(
      --include='/logs.json.txt'
      --include='/*.yaml'
      --include='/.hydra/***'
    )
  fi
  filters+=(--exclude='*')

  # Checkpoints are effectively incompressible, so SSH compression stays off.
  # New files stream directly; --append-verify resumes interrupted copies and
  # validates the completed file instead of retransferring it from byte zero.
  rsync_args=(
    -a
    --append-verify
    --protect-args
    --human-readable
    --info=progress2,stats1
    -e "ssh -T -o BatchMode=yes -o Compression=no"
  )

  echo "==> [$dest_name] transferring from $DATA_HOST:$run_dir"
  if rsync "${rsync_args[@]}" "${filters[@]}" \
      "$DATA_HOST:$run_dir/" "$dest_dir/"; then
    printf '%s\n' "$DATA_HOST:$run_dir" > "$dest_dir/.psc-source"
    if ((COPY_EVAL_SCRIPTS)); then
      stage_eval_scripts "$run_dir" "$dest_dir"
    fi
    echo "==> [$dest_name] complete"
  else
    echo "!! [$dest_name] transfer failed (rerun to resume)" >&2
    return 1
  fi
}

# Static worker queues cap concurrent large transfers without spawning a new
# SSH connection per checkpoint. Two workers is a conservative fast default.
worker() {
  local worker_id="$1" index rc=0
  for ((index = worker_id; index < ${#ACTIVE_RUNS[@]}; index += JOBS)); do
    sync_run "$index" || rc=1
  done
  return "$rc"
}

declare -a worker_pids=()
worker_count="$JOBS"
((worker_count > ${#ACTIVE_RUNS[@]})) && worker_count="${#ACTIVE_RUNS[@]}"
for ((worker_id = 0; worker_id < worker_count; worker_id++)); do
  worker "$worker_id" &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || failed=1
done

if ((failed)); then
  die "one or more transfers failed; rerun the same command to resume"
fi

echo
echo "==> Done. Synced ${#ACTIVE_RUNS[@]} run(s) into $LOCAL_DEST"
