#!/bin/bash
# Train articubot (GMM cross-displacement high-level policy) on
# frankaPushBlock (real-world Franka push-block demos, 50 demos) using
# UVD subgoals (goal_gripper_pcd_uvd) from the EXTRA_KEYPOINTS_uvd tree.
#
# Local run on GPU 1, 100 epochs.

set -euo pipefail
set -x

REPO_DIR="/home/theyanesh/worktrees/theya_high_level_lfd3d"
cd "${REPO_DIR}"

# --- resume from checkpoint ----------------------------------------------
# Resumes full Lightning state (weights + optimizer + scheduler + epoch) via
# the +resume_from override in scripts/train.py. Defaults to the last intact
# periodic checkpoint of the 2026-08-14/11-32-02 run (epoch 34).
# Set RESUME_CKPT="" to train from scratch:
#   RESUME_CKPT="" bash this_script.sh
RESUME_CKPT="${REPO_DIR}/logs/train_frankaPushBlock_UVD_subgoals/2026-08-14/11-32-02/checkpoints/periodic-epoch=epoch=34.ckpt"

RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    echo "[resume] resuming training from ${RESUME_CKPT}"
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    # Hydra-level single quotes are required: the checkpoint filename contains
    # '=' (periodic-epoch=epoch=34.ckpt), which otherwise breaks the override
    # grammar ("mismatched input '=' expecting <EOF>").
    RESUME_ARGS=("+resume_from='${RESUME_CKPT}'")
else
    echo "[resume] RESUME_CKPT empty -> training from scratch"
fi

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
pixi run python scripts/train.py \
    model=articubot \
    dataset=frankaPushBlock \
    model.use_rgb=False \
    model.in_channels=4 \
    training.batch_size=64 \
    training.epochs=100 \
    wandb.entity=humantorobot \
    "hydra.run.dir=logs/train_frankaPushBlock_UVD_subgoals/$(date +%Y-%m-%d/%H-%M-%S)" \
    +dataset.goal_source=uvd \
    +dataset.extra_goals_dir=/data/theya/data/uncertainity_subgoal/EXTRA_KEYPOINTS_uvd/franka_push_block_mimicgen_npz \
    +dataset.use_weighted_sampler=True \
    +dataset.transition_p=0.5 \
    +dataset.transition_radius=5 \
    +dataset.transition_label_swap=True \
    +dataset.transition_swap_p_max=0.5 \
    "resources.gpus=[1]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
