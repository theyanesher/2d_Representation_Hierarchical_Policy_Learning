#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 12:00:00 # 12-hour budget
#SBATCH --job-name coffee-prep-d1-wca-100demo-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo version of the COFFEE_PREPERATION_D1 GMM weighted-cross-attention baseline.
#
# Train flow-matching DiT low-level policy on COFFEE_PREPERATION_D1 with GMM
# weighted cross-attention. Uses the FIRST NUM_DEMOS demos (demo_0.h5 ..
# demo_(N-1).h5) from the full 1000-trajectory GMM dataset on /ocean. Only the
# requested demos are rsynced to node-local scratch — the rest stay on /ocean.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume / epoch cap --------------------------------------------------
# Relaunch (resume full state: model + EMA + optimizer + epoch counter) from
# this checkpoint. Set RESUME_CKPT="" to train from scratch instead.
RESUME_CKPT="${RESUME_CKPT:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.06.18/21.18.43_groot_GMM_WCA_100demo_dinov2_Coffee_Preperation_D1_coffee_preperation_gmm_goal/checkpoints/epoch_80.ckpt}"
# Stop training once it reaches this ABSOLUTE epoch.
TARGET_EPOCH="${TARGET_EPOCH:-100}"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide. PSC's SLURM doesn't always pre-create
# that subtree, so we force-create it ourselves when SLURM_JOB_ID is set. Fall
# back to $LOCAL or /tmp only if no per-job dir exists.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Low_Level_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] files  : demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5  (${NUM_DEMOS} files)"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Generate exactly the demo filenames we want and feed to xargs.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Translate "resume from <ckpt> and stop at epoch TARGET_EPOCH" into hydra
# overrides. The training loop is `for _ in range(training.num_epochs)` with
# self.epoch RESTORED from the checkpoint, so num_epochs is a *relative* count
# of epochs THIS run executes (NOT an absolute target). When resuming we derive
# it from the checkpoint's epoch number so the run ends — and writes a final
# checkpoint — exactly at TARGET_EPOCH. The '+' on resume_ckpt_path is required
# because that key isn't in the base config.
RESUME_ARGS=()
EPOCH_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    RESUME_ARGS=(training.resume=true "+training.resume_ckpt_path=${RESUME_CKPT}")
    ckpt_base="$(basename "${RESUME_CKPT}")"
    if [[ "${ckpt_base}" =~ epoch_([0-9]+) ]]; then
        resumed_epoch="${BASH_REMATCH[1]}"
        # +1 so the loop reaches AND checkpoints epoch == TARGET_EPOCH
        # (checkpoints save on the pre-increment epoch value).
        rel_epochs=$(( TARGET_EPOCH - resumed_epoch + 1 ))
        if [ "${rel_epochs}" -le 0 ]; then
            echo "[resume] ERROR: resumed epoch ${resumed_epoch} >= TARGET_EPOCH ${TARGET_EPOCH}; nothing to train." >&2
            exit 1
        fi
        EPOCH_ARGS=(training.num_epochs="${rel_epochs}")
        echo "[resume] resuming from epoch ${resumed_epoch}; TARGET_EPOCH=${TARGET_EPOCH} -> training.num_epochs=${rel_epochs} (trains epochs ${resumed_epoch}..${TARGET_EPOCH}, saves epoch_${TARGET_EPOCH})"
    else
        echo "[resume] WARN: could not parse epoch number from '${ckpt_base}'; leaving training.num_epochs at config default." >&2
    fi
else
    EPOCH_ARGS=(training.num_epochs="${TARGET_EPOCH}")
    echo "[resume] RESUME_CKPT empty -> training from scratch for ${TARGET_EPOCH} epochs"
fi

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
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=128 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1 \
    name=groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1 \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
