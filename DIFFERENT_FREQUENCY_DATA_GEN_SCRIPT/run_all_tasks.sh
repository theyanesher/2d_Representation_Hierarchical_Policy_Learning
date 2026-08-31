#!/usr/bin/env bash
# Build the sampling-rate ablation for ALL tasks.
#
# build_rate_ablation.py is deliberately per-task (one task, six rate arms, three
# stages). This launcher drives it across the three tasks with the right
# parallelism for each stage, in the same style as Generate_Final_Dataset_*.sh:
#
#   stages resim+stride : CPU-only and fast (~10 min for all three). Run all
#                         three concurrently; 3 x 16 workers on 128 threads.
#   stage  render       : GPU-bound and slow (~44 h total). Only 2 GPUs, so run
#                         two tasks at a time, one per GPU, then the third.
#
# Every stage is resumable -- re-running skips work whose output already exists,
# so an interrupted render picks up at the arm it died on.
#
# Usage:
#   ./run_all_tasks.sh                 # everything
#   ./run_all_tasks.sh resim,stride    # states only, decide on rendering later
#   ./run_all_tasks.sh render

set -uo pipefail   # NOT -e: background jobs are reaped explicitly below

STAGES="${1:-resim,stride,render}"

ROOT="/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning"
PY="${ROOT}/.pixi/envs/eval/bin/python"
SCRIPT="${ROOT}/DIFFERENT_FREQUENCY_DATA_GEN_SCRIPT/build_rate_ablation.py"
OUT_ROOT="${OUT_ROOT:-/data/theya/data/rate_ablation}"

TASKS=(hammer_cleanup_d1 coffee_preparation_d1 kitchen_d1)
GPUS=(0 1)

N_DEMOS="${N_DEMOS:-140}"           # ~77% survive replay -> ~108 kept
POOL_SIZE="${POOL_SIZE:-16}"        # CPU workers for re-simulation
RENDER_POOL="${RENDER_POOL:-6}"     # VRAM-bound, ~6 per 24 GB GPU
RATES="${RATES:-5 10 20 50 100 250}"

mkdir -p "${ROOT}/logs" "${OUT_ROOT}"
cd "${ROOT}"

run_stage() {   # $1=task  $2=gpu  $3=stages  $4=logsuffix
  "${PY}" "${SCRIPT}" \
      --task "$1" --gpu "$2" --stages "$3" \
      --out_root "${OUT_ROOT}" \
      --rates ${RATES} \
      --n "${N_DEMOS}" \
      --pool_size "${POOL_SIZE}" \
      --render_pool_size "${RENDER_POOL}" \
      > "${ROOT}/logs/rate_ablation_$1_$4.log" 2>&1
}

fail=0

# --------------------------------------------------------------------------- #
# States: all three tasks at once (CPU only, no GPU contention)
# --------------------------------------------------------------------------- #
if [[ "${STAGES}" == *resim* || "${STAGES}" == *stride* ]]; then
  SUB=$(echo "${STAGES}" | tr ',' '\n' | grep -E '^(resim|stride)$' | paste -sd,)
  echo "=== stages [${SUB}] : all ${#TASKS[@]} tasks in parallel ==="
  pids=()
  for t in "${TASKS[@]}"; do
    run_stage "$t" 0 "${SUB}" states &
    pids+=($!); echo "  launched $t (pid $!)"
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  [FAIL] ${TASKS[$i]} -- see logs/rate_ablation_${TASKS[$i]}_states.log"; fail=1; }
  done
  echo "=== states done ==="
fi

# --------------------------------------------------------------------------- #
# Render: two at a time, one GPU each
# --------------------------------------------------------------------------- #
if [[ "${STAGES}" == *render* ]]; then
  # Tasks are dealt round-robin to GPUs UP FRONT, and each GPU then works its
  # own queue sequentially while all GPUs run concurrently.
  #
  # The obvious alternative -- run 2 tasks, wait, run the 3rd -- leaves a GPU
  # completely idle during the final batch. With 3 tasks that wasted ~9-14 h.
  # Two tasks on ONE GPU simultaneously is not an option either: that would be
  # 2 x RENDER_POOL workers against 24 GB.
  echo "=== stage [render] : ${#TASKS[@]} tasks over ${#GPUS[@]} GPUs, one queue per GPU ==="
  pids=(); owner=()
  for g in "${!GPUS[@]}"; do
    queue=()
    for (( i=g; i<${#TASKS[@]}; i+=${#GPUS[@]} )); do queue+=("${TASKS[$i]}"); done
    (( ${#queue[@]} )) || continue
    echo "  GPU ${GPUS[$g]} queue: ${queue[*]}"
    (
      for t in "${queue[@]}"; do
        echo "  [GPU ${GPUS[$g]}] starting $t"
        run_stage "$t" "${GPUS[$g]}" render render \
          || { echo "  [FAIL] $t"; exit 1; }
      done
    ) &
    pids+=($!); owner+=("GPU ${GPUS[$g]} (${queue[*]})")
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { echo "  [FAIL] ${owner[$j]} -- see logs/rate_ablation_*_render.log"; fail=1; }
  done
  echo "=== render done ==="
fi

echo
if (( fail )); then
  echo "FINISHED WITH FAILURES -- check ${ROOT}/logs/rate_ablation_*.log"
else
  echo "All tasks complete."
fi
echo "  base : ${OUT_ROOT}/base_500hz"
echo "  arms : ${OUT_ROOT}/arms"
echo "  npz  : ${OUT_ROOT}/npz"
exit "${fail}"
