#!/usr/bin/env bash
# Evaluate every Approach 1 run (goal_gripper AND wca_rope_goals) under
# approach1_baselines_best_ckpt/.
#
#   ./run_all_evals.sh                 # launch
#   ./run_all_evals.sh --dry-run       # print the queue + slot layout, run nothing
#   ./run_all_evals.sh --filter kitchen   # only runs whose path matches (regex)
#
# Discovers <Task_D1>/<run>/ dirs that hold checkpoints/ + .hydra/config.yaml
# (i.e. whatever ./all_download.sh has fetched so far; Coffee joins the queue
# automatically once its folder is complete), links eval.sh into each, asks
# eval.sh --list which LL family it is (goal_gripper -> GHOST evaluator, rope ->
# RoPE GMM evaluator), and queues ONE JOB PER (run, checkpoint):
#     bash <run>/eval.sh -c <ckpt>        (all default seeds)
# Queue order is run-major (Hammer -> Kitchen -> Coffee), oldest epoch first.
#
# Four jobs run at once: two slots pinned to GPU 0, two to GPU 1 (see
# SLOT_GPU). Each slot pulls the next job off a shared queue as soon as it
# frees up. Each job uses NUM_ENVS MuJoCo workers (default 8 -> up to 32
# workers total); lower it with NUM_ENVS=4 ./run_all_evals.sh if RAM/CPU gets
# tight. Jobs from the same run never collide: every checkpoint writes to its
# own <run>/evaluations/<ckpt>/ folder.
#
# Resumable: eval.sh skips (checkpoint, seed) pairs that already have all
# episodes and resumes partial ones, so re-running this script only does
# what is missing (and picks up checkpoints that arrived since).
#
# Logs: _eval_logs/<task>__<run>__<ckpt>.log per job, _eval_logs/launcher.log timeline.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/_eval_logs"
DRY_RUN=0
FILTER=""
while (($#)); do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    --filter)     FILTER="$2"; shift 2 ;;
    -h|--help)    awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
done

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export NUM_ENVS="${NUM_ENVS:-8}"

# slot -> GPU  (4 concurrent jobs, two per GPU)
SLOT_GPU=(0 0 1 1)

# ---- discover runs (task dir order: Hammer -> Kitchen -> Coffee) -------------
RUNS=()
RUN_FAMILY=()
SEEN=()
for task_dir in "$ROOT"/Hammer_CleanUp_D1 "$ROOT"/Kitchen_D1 "$ROOT"/Coffee_Prep_D1 "$ROOT"/*_D1; do
  [[ -d "$task_dir" ]] || continue
  for run_dir in "$task_dir"/*/; do
    run_dir="${run_dir%/}"
    [[ -d "$run_dir/checkpoints" && -f "$run_dir/.hydra/config.yaml" ]] || continue
    rel="${run_dir#"$ROOT"/}"
    # de-duplicate (the *_D1 glob re-visits the explicitly listed dirs)
    for r in "${SEEN[@]}"; do [[ "$r" == "$rel" ]] && continue 2; done
    SEEN+=("$rel")
    [[ -z "$FILTER" || "$rel" =~ $FILTER ]] || continue
    ls "$run_dir"/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "skip (no ckpts yet): $rel" >&2; continue; }
    # Every run gets a symlink to the shared launcher (harmless, even in dry-run).
    [[ -e "$run_dir/eval.sh" ]] || ln -s ../../eval.sh "$run_dir/eval.sh"
    # eval.sh picks the evaluator per LL family (goal_gripper -> GHOST,
    # rope -> RoPE GMM, gmm -> GMM parallel); ask it whether it can serve this run.
    if ! out="$(bash "$run_dir/eval.sh" --list 2>&1)"; then
      echo "skip (eval.sh cannot serve it): $rel" >&2
      echo "$out" | grep "ERROR" | sed 's/^/    /' >&2
      continue
    fi
    fam="$(echo "$out" | sed -n 's/^\[derived\] LL family   : \([a-z_]*\).*/\1/p')"
    RUNS+=("$rel")
    RUN_FAMILY+=("$fam")
  done
done
((${#RUNS[@]})) || { echo "error: no servable runs with checkpoints/ + .hydra/config.yaml under $ROOT" >&2; exit 2; }

# ---- expand to (run, checkpoint) jobs, oldest epoch first -------------------
ckpt_epoch() { local n="${1##*/}"; n="${n%.ckpt}"; [[ "$n" =~ ([0-9]+)[^0-9]*$ ]] && echo "${BASH_REMATCH[1]}" || echo -1; }
JOBS=()   # "<run>|<ckpt file name>|<family>"
for ((ri = 0; ri < ${#RUNS[@]}; ri++)); do
  r="${RUNS[$ri]}"; fam="${RUN_FAMILY[$ri]}"
  while IFS= read -r ck; do
    [[ -n "$ck" ]] && JOBS+=("$r|$ck|$fam")
  done < <(for f in "$ROOT/$r"/checkpoints/*.ckpt; do printf '%s\t%s\n' "$(ckpt_epoch "$f")" "$(basename "$f")"; done | sort -n -k1,1 | cut -f2-)
done

if ((DRY_RUN)); then
  echo "Queue (start order), ${#JOBS[@]} jobs over ${#RUNS[@]} runs, ${#SLOT_GPU[@]} concurrent, NUM_ENVS=$NUM_ENVS:"
  for ((i = 0; i < ${#JOBS[@]}; i++)); do
    IFS='|' read -r jr jc jf <<<"${JOBS[$i]}"
    printf '  %2d. %-62s %-14s [%s]\n' "$((i + 1))" "$jr" "$jc" "$jf"
  done
  echo "Slots: $(for ((s = 0; s < ${#SLOT_GPU[@]}; s++)); do printf 'slot%d->GPU%s ' "$s" "${SLOT_GPU[$s]}"; done)"
  echo "Each job: bash <run>/eval.sh -c <ckpt>"
  exit 0
fi

mkdir -p "$LOGS"
QUEUE="$(mktemp "$LOGS/.queue.XXXXXX")"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
trap 'rm -f "$QUEUE" "$QUEUE.lock"' EXIT

pop() {  # atomically remove and print the first queued job ("" when empty)
  (
    flock 9
    head -n1 "$QUEUE"
    sed -i '1d' "$QUEUE"
  ) 9>"$QUEUE.lock"
}

worker() {
  local slot="$1" gpu="$2" job run ck rc
  while job="$(pop)"; [[ -n "$job" ]]; do
    IFS='|' read -r run ck _fam <<<"$job"
    local log="$LOGS/${run//\//__}__${ck%.ckpt}.log"
    echo "##### START $run  $ck  slot=$slot gpu=$gpu  $(date) #####" | tee -a "$log" "$LOGS/launcher.log"
    CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/$run/eval.sh" -c "$ck" >>"$log" 2>&1
    rc=$?
    echo "##### END   $run  $ck  rc=$rc  $(date) #####" | tee -a "$log" "$LOGS/launcher.log"
  done
}

echo "##### LAUNCH ${#JOBS[@]} jobs (${#RUNS[@]} runs), ${#SLOT_GPU[@]} slots, NUM_ENVS=$NUM_ENVS  $(date) #####" | tee -a "$LOGS/launcher.log"
for ((s = 0; s < ${#SLOT_GPU[@]}; s++)); do
  worker "$s" "${SLOT_GPU[$s]}" &
done
wait
echo "##### ALL DONE  $(date) #####" | tee -a "$LOGS/launcher.log"
