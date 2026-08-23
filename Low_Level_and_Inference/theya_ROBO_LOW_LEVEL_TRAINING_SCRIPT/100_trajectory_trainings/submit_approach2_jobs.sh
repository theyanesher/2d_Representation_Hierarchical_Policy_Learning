#!/bin/bash
# Submit the 12 Approach2 (GMM-as-auxiliary-loss) jobs across
# HAMMER_CLEANUP_D1 / KITCHEN_D1 / COFFEE_PREPERATION_D1, and log the
# resulting SLURM job IDs so they're easy to track afterwards.
#
# Usage:
#   ./submit_approach2_jobs.sh                # submit all 12
#   ./submit_approach2_jobs.sh HAMMER_CLEANUP_D1_TASK   # only this task's 4
#
# Job IDs (with task/goal_source/timestamp) are appended to
# approach2_job_log.tsv in this directory. Check status any time with:
#   squeue -u "$USER" --format="%.10i %.45j %.8T %.10M %.6D %R"
#   sacct -j <id1>,<id2>,... --format=JobID,JobName%40,State,Elapsed,ExitCode
#   tail -f ../logs/<job-name>_job_<id>.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/approach2_job_log.tsv"
FILTER_TASK="${1:-}"

SCRIPTS=(
    "HAMMER_CLEANUP_D1_TASK/Approach2/hammercleanupD1_approach2_awe.sh"
    "HAMMER_CLEANUP_D1_TASK/Approach2/hammercleanupD1_approach2_uvd.sh"
    "HAMMER_CLEANUP_D1_TASK/Approach2/hammercleanupD1_approach2_mix_bspline_bspline_greville.sh"
    "HAMMER_CLEANUP_D1_TASK/Approach2/hammercleanupD1_approach2_mix_gripper_heuristic_orientation_heuristic.sh"
    "KITCHEN_D1_TASK/Approach2/kitchenD1_approach2_awe.sh"
    "KITCHEN_D1_TASK/Approach2/kitchenD1_approach2_uvd.sh"
    "KITCHEN_D1_TASK/Approach2/kitchenD1_approach2_mix_bspline_bspline_greville.sh"
    "KITCHEN_D1_TASK/Approach2/kitchenD1_approach2_mix_gripper_heuristic_orientation_heuristic.sh"
    "COFFEE_PREPERATION_D1_TASK/Approach2/coffee_preperationD1_approach2_awe.sh"
    "COFFEE_PREPERATION_D1_TASK/Approach2/coffee_preperationD1_approach2_uvd.sh"
    "COFFEE_PREPERATION_D1_TASK/Approach2/coffee_preperationD1_approach2_mix_bspline_bspline_greville.sh"
    "COFFEE_PREPERATION_D1_TASK/Approach2/coffee_preperationD1_approach2_mix_gripper_heuristic_orientation_heuristic.sh"
)

if [ ! -f "${LOG_FILE}" ]; then
    printf "timestamp\tjob_id\tscript\n" > "${LOG_FILE}"
fi

echo "job_id  script"
for rel in "${SCRIPTS[@]}"; do
    if [ -n "${FILTER_TASK}" ] && [[ "${rel}" != "${FILTER_TASK}"* ]]; then
        continue
    fi
    job_id=$(sbatch --parsable "${SCRIPT_DIR}/${rel}")
    printf "%s\t%s\t%s\n" "$(date -Is)" "${job_id}" "${rel}" >> "${LOG_FILE}"
    printf "%-8s%s\n" "${job_id}" "${rel}"
done

echo
echo "Logged to ${LOG_FILE}"
echo "Track with: squeue -u \"\$USER\" --format=\"%.10i %.45j %.8T %.10M %.6D %R\""
