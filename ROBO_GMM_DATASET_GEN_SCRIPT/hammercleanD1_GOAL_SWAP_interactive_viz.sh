#!/bin/bash
# Interactive GMM visualization for HAMMER_CLEANUP_D1 — ONE demo only, using
# the NEW GOAL_SWAP high-level model (same checkpoint as
# hammercleanD1_GOAL_SWAP.sh: 100-demo GOAL_SWAP run, epoch=59 best val/rmse).
#
# Viewer/debug variant of hammercleanD1_GOAL_SWAP.sh. Runs the high-level GMM
# model on a SINGLE demo (--max_files 1) and launches the viser viewer showing:
#   --visualize               per-step anchors (weight-colored) + sampled goal
#   --visualize_all_gmm_goals every GMM component goal, faded by weight (green)
#   --visualize_gmm_modes     the halo-collapsed modes (orange/cyan/yellow by
#                             mode index) + per-mode weights in the GUI
#
# It does NOT stage to /local and does NOT ship results back to /ocean — it
# reads the source npz straight from /ocean and writes the (single) h5 to a
# small viz output dir. The point is to LOOK at the GMM goals/modes.
#
# --- HOW TO RUN (interactive, needs a GPU) -------------------------------
#   1. Grab an interactive H100 session on the ROBO partition, e.g.:
#        interact -p ROBO --gres=gpu:h100:1 -n 1 --cpus-per-task=12 -t 02:00:00
#   2. From your LAPTOP, forward the viser port so you can open the viewer:
#        ssh -L 8080:<compute-node-hostname>:8080 pbhowal@bridges2.psc.edu
#      (get <compute-node-hostname> from `squeue -u pbhowal` / `hostname`)
#   3. Run this script inside the interactive session:
#        bash hammercleanD1_GOAL_SWAP_interactive_viz.sh
#      (visualize demo N:  START_DEMO=N bash hammercleanD1_GOAL_SWAP_interactive_viz.sh)
#   4. Open http://localhost:8080 in your laptop browser.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths (same as hammercleanD1_GOAL_SWAP.sh) --------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/HAMMER_CLEANUP_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
CKPT_PATH="${REPO_DIR}/logs/train_HammerCleanup_D1_GOAL_SWAP_100demo/2026-07-19/20-45-59/checkpoints/epoch=59-step=12180-val/rmse=0.055.ckpt"
# Small viz-only output dir (one h5 lands here). Override with VIZ_OUT_DIR.
VIZ_OUT_DIR="${VIZ_OUT_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/GMM_VIZ_OUTPUT/HAMMER_CLEANUP_D1_GOAL_SWAP}"

# Which demo to visualize (0-based). Override with START_DEMO=N.
START_DEMO="${START_DEMO:-0}"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

mkdir -p "${VIZ_OUT_DIR}"
cd "${REPO_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \
    --dataset_dir "${SRC_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --gmm_output_dir "${VIZ_OUT_DIR}" \
    --start_demo "${START_DEMO}" \
    --max_files 1 \
    --batch_size 32 \
    --visualize \
    --visualize_all_gmm_goals \
    --visualize_gmm_modes \
    --mode_radius 0.03 \
    --max_modes 3
