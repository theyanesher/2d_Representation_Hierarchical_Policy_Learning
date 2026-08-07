#!/bin/bash
# Interactive GMM visualization for PushT_Task — a few demos, using the
# GOAL_SWAP high-level model trained on all 206 PushT demos
# (train_PushT_Task_GOAL_SWAP 2026-07-19 run, epoch=89 best val
# rmse_and_std_combi checkpoint).
#
# Runs the high-level GMM model on MAX_FILES demos (default 3, starting at
# START_DEMO) and launches the viser viewer per demo showing the classic view:
#   --visualize               per-step anchors (weight-colored) + sampled goal
#   --visualize_all_gmm_goals every GMM component goal in green, faded by weight
# Ctrl+C closes the current demo's viewer and ADVANCES to the next demo.
#
# --push_t: PushT npz has no camera intrinsics/extrinsics (and all-zero depth),
# so the generator reads/writes no camera keys — the GMM forward pass and the
# viser viewer only need point_cloud / gripper_pcd / goal_gripper_pcd.
#
# It does NOT stage to /local and does NOT ship results back anywhere useful —
# it reads the source npz straight from /ocean and writes the (few) h5 files to
# a small viz output dir. The point is to LOOK at the GMM goals/modes.
#
# --- HOW TO RUN (interactive, needs a GPU) -------------------------------
#   1. Grab an interactive H100 session on the ROBO partition, e.g.:
#        interact -p ROBO --gres=gpu:h100:1 -n 1 --cpus-per-task=12 -t 02:00:00
#   2. From your LAPTOP, forward the viser port so you can open the viewer:
#        ssh -L 8080:<compute-node-hostname>:8080 pbhowal@bridges2.psc.edu
#      (get <compute-node-hostname> from `squeue -u pbhowal` / `hostname`)
#   3. Run this script inside the interactive session:
#        bash pushT_GOAL_SWAP_interactive_viz.sh
#      (other demos:      START_DEMO=17 bash pushT_GOAL_SWAP_interactive_viz.sh)
#      (single demo only: MAX_FILES=1  bash pushT_GOAL_SWAP_interactive_viz.sh)
#   4. Open http://localhost:8080 in your laptop browser.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/PushT_Task"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
CKPT_PATH="${REPO_DIR}/logs/train_PushT_Task_GOAL_SWAP/2026-07-19/23-17-19/checkpoints/epoch=89-step=16200-val/rmse_and_std_combi=0.060.ckpt"
# Small viz-only output dir (a few h5s land here). Override with VIZ_OUT_DIR.
VIZ_OUT_DIR="${VIZ_OUT_DIR:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/GMM_VIZ_OUTPUT/PUSH_T_TASK_GOAL_SWAP}"

# Which demos to visualize (0-based start + count). Ctrl+C advances demos.
START_DEMO="${START_DEMO:-0}"
MAX_FILES="${MAX_FILES:-3}"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

mkdir -p "${VIZ_OUT_DIR}"
cd "${REPO_DIR}"

# Classic view: weight-colored anchors + all GMM component goals in green,
# faded by weight. To ALSO overlay the halo-collapsed modes (each mode's
# keypoints in a distinct color: orange/cyan/yellow by index), add:
#   --visualize_gmm_modes --mode_radius 0.02 --max_modes 3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \
    --dataset_dir "${SRC_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --gmm_output_dir "${VIZ_OUT_DIR}" \
    --push_t \
    --start_demo "${START_DEMO}" \
    --max_files "${MAX_FILES}" \
    --batch_size 32 \
    --visualize \
    --visualize_all_gmm_goals
