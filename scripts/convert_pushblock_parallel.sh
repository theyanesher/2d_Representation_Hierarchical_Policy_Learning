#!/usr/bin/env bash
# Shard the pushblock conversion across several worker processes.
#
# WHY THIS HELPS. Profiling one episode (349 frames, 17s wall) puts the time in
# stages that a single process cannot spread over cores:
#     zlib.compress (np.savez_compressed)  5.35s   single-threaded
#     decode_video_frames (6 streams)      4.38s   barely threaded
#     resize / remap (OpenCV)              1.90s   already multi-threaded
# One process peaks around 23 of the 64 cores and leaves the rest idle. Episodes
# are fully independent -- each writes its own demo_<i>/ directory -- so running
# N processes over disjoint episode subsets is the whole trick. No converter
# change; it is just --episodes with a different list per worker.
#
# THREAD CAP. OpenCV grabs all 64 cores per process by default, so N workers
# would oversubscribe badly (N x 64 threads fighting over 64 cores). Each worker
# is capped to CV_THREADS instead.
#
# MEMORY. decode_video_frames holds every frame of all 6 streams in RAM: ~5.4GB
# peak per worker for a 349-frame episode, more for longer ones. Budget roughly
# 8GB per worker and check `free -g` before raising WORKERS.
#
# REPRODUCIBILITY CAVEAT. The converter seeds one RNG per PROCESS
# (np.random.default_rng(--seed) in main) and threads it through every episode
# in turn, so which 4500 points `point_cloud` keeps for a given episode depends
# on how many frames were converted before it in that process. Sharding changes
# that grouping, so clouds will differ from a serial run -- a different random
# subsample of the same cropped cloud, not a worse or wrong one. Nothing else in
# the npz is affected. If you need shard-independent output, seed per episode in
# the converter instead: np.random.default_rng(args.seed + ep_idx).
#
# Usage:
#   bash scripts/convert_pushblock_parallel.sh                 # 8 workers, all episodes
#   WORKERS=16 bash scripts/convert_pushblock_parallel.sh
#   N_EPISODES=20 WORKERS=4 bash scripts/convert_pushblock_parallel.sh
#   OUTPUT_DIR=/somewhere/else bash scripts/convert_pushblock_parallel.sh
set -euo pipefail

# SCRIPT_DIR locates the sibling wrapper and must NOT follow REPO_ROOT: the two
# can differ (REPO_ROOT points at the checkout holding the python converter,
# which may not be the tree this script lives in).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKERS="${WORKERS:-8}"
CV_THREADS="${CV_THREADS:-4}"
LEROBOT_DIR="${LEROBOT_DIR:-/home/madhavan/uncertain_subgoal/lerobot/data/pushblock}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/madhavan/uncertain_subgoal/lerobot/data/pushblock_npz}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/_convert_logs}"

# Episode count from the dataset unless overridden.
if [[ -z "${N_EPISODES:-}" ]]; then
    N_EPISODES=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" \
                 "${LEROBOT_DIR}/meta/info.json")
fi

mkdir -p "${LOG_DIR}"
echo "[parallel] ${N_EPISODES} episodes over ${WORKERS} workers (${CV_THREADS} OpenCV threads each)"
echo "[parallel] output ${OUTPUT_DIR}"
echo "[parallel] logs   ${LOG_DIR}"

# Round-robin, not contiguous blocks: episode lengths vary by ~2x, so striping
# keeps the workers finishing at about the same time.
declare -a SHARD
for ((e = 0; e < N_EPISODES; e++)); do
    w=$((e % WORKERS))
    SHARD[w]="${SHARD[w]:-} ${e}"
done

pids=()
for ((w = 0; w < WORKERS; w++)); do
    [[ -z "${SHARD[w]:-}" ]] && continue
    # shellcheck disable=SC2086
    OPENCV_FOR_THREADS_NUM="${CV_THREADS}" \
    OMP_NUM_THREADS="${CV_THREADS}" \
    REPO_ROOT="${REPO_ROOT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/convert_pushblock.sh" \
            --episodes ${SHARD[w]} \
            >"${LOG_DIR}/worker_${w}.log" 2>&1 &
    pids+=($!)
    echo "[parallel] worker ${w} (pid $!):$(echo "${SHARD[w]}" | cut -c1-60)..."
done

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
        echo "[parallel] WORKER ${i} FAILED -- see ${LOG_DIR}/worker_${i}.log" >&2
        rc=1
    fi
done

if [[ ${rc} -eq 0 ]]; then
    got=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'demo_*' -type d | wc -l)
    echo "[parallel] done: ${got}/${N_EPISODES} demo dirs in ${OUTPUT_DIR}"
    [[ "${got}" -ne "${N_EPISODES}" ]] && { echo "[parallel] COUNT MISMATCH" >&2; rc=1; }
fi
exit ${rc}
