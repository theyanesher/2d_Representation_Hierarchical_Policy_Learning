#!/bin/bash
# Run all four ablation GMM-prediction generations sequentially in the current
# interactive GPU session:  bash run_all_interactive.sh
#
# Order: PushT first (small, ~1.3GB staging, finishes fast) then the two
# kitchen runs (~22GB staging, shared between them — the second kitchen run
# reuses the already-staged inputs on /local and skips straight to generation).
#
# Safe to interrupt and re-run: outputs go directly to /ocean and finished
# demos are skipped, so a rerun continues where it stopped. If the session
# clock runs out mid-way, just re-run this in the next session (only the
# /local input staging is redone).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in \
    pushT_noswap_interactive.sh \
    pushT_sigmoid_interactive.sh \
    kitchenD1_noswap_interactive.sh \
    kitchenD1_sigmoid_interactive.sh; do
    echo "=================================================================="
    echo "[run_all] starting ${s}  ($(date))"
    echo "=================================================================="
    bash "${HERE}/${s}"
done
echo "[run_all] all four generations complete ($(date))"
