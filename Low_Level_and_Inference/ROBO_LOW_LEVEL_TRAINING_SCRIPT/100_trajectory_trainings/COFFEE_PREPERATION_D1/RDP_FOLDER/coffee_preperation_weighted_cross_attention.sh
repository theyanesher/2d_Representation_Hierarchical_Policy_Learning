#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 4:00:00 # resume budget: 14 remaining epochs (86..99) at ~7.9
                   # min/epoch (orig run: 86 epochs in ~11.5h) ~= 2h + margin.
                   # Use 12:00:00 for a fresh run (RESUME_CKPT="").
#SBATCH --job-name coffee-prep-d1-wca-100demo-dinov2-rdp
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/RDP_FOLDER/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/RDP_FOLDER/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# RDP version of the COFFEE_PREPERATION_D1 GMM weighted-cross-attention baseline.
#
# Same flow-matching DiT + DINOv2 + WCA setup as
# ../coffee_preperation_weighted_cross_attention.sh, but the GMM conditioning
# (gmm_all_goals / gmm_all_weights) comes from the RDP-goal-trained high-level
# model: it is read PER FRAME from the prediction npz tree
# (EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1_GMM_PRED, keys gmm_all_goals_rdp /
# gmm_all_weights_rdp) instead of the h5 files. The h5 files are read-only and
# their default gmm keys are never loaded (dataset arg gmm_pred_npz_dir).
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh   (needs predictions for those demos!)

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1"
PRED_SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/COFFEE_PREPERATION_D1_GMM_PRED"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Low_Level_${NUM_DEMOS}demo"
DEST_PRED_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_GMM_PRED_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] h5 source  : ${SRC_DATA_DIR}"
echo "[stage] pred source: ${PRED_SRC_DIR}"
echo "[stage] h5 dest    : ${DEST_DATA_DIR}"
echo "[stage] pred dest  : ${DEST_PRED_DIR}"
mkdir -p "${DEST_DATA_DIR}" "${DEST_PRED_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR PRED_SRC_DIR DEST_PRED_DIR

# h5 episode files.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

# RDP prediction npz tree (per-frame demo_N/<t>.npz).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${PRED_SRC_DIR}/$1" "${DEST_PRED_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
pred_count=$(
    find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} h5 files, ${pred_count} pred demo dirs with npz."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} h5 files staged, got ${staged_count}." >&2
    exit 1
fi
if [ "${pred_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: only ${pred_count}/${NUM_DEMOS} prediction demo dirs have npz — run the RDP_DATAGEN job first." >&2
    exit 1
fi

# --- resume from checkpoint ----------------------------------------------
# Default: epoch_85.ckpt of the 2026.07.08 run (stopped at epoch 85 of 100
# during the July-8 quota/walltime crunch). training.num_epochs=100 below is
# an ABSOLUTE stop — the resumed run trains epochs 86..99 and terminates; it
# can NOT run past 100. Set RESUME_CKPT="" (and -t 12h) for a fresh run:
#   RESUME_CKPT="" sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.07.08/02.42.29_groot_GMM_WCA_100demo_dinov2_Coffee_Preperation_D1_RDP_coffee_preperation_gmm_goal/checkpoints/epoch_85.ckpt}"

# Tag the W&B run so the resume leg is distinguishable from the original.
RESUME_ARGS=()
RESUME_TAG=""
if [ -n "${RESUME_CKPT}" ]; then
    echo "[resume] resuming training from ${RESUME_CKPT}"
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    RESUME_ARGS=(training.resume=true "+training.resume_ckpt_path=${RESUME_CKPT}")
    RESUME_TAG="_resumeE$(basename "${RESUME_CKPT}" .ckpt | grep -oE '[0-9]+' || echo X)"
else
    echo "[resume] RESUME_CKPT empty -> training from scratch"
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_gmm_goal \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.gmm_pred_npz_dir="${DEST_PRED_DIR}" \
    +task.dataset.gmm_pred_key_suffix=rdp \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_RDP${RESUME_TAG} \
    name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_RDP${RESUME_TAG} \
    training.checkpoint_every=5 \
    training.num_epochs=100 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
