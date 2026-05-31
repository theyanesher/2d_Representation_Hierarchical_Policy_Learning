#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name mug-cleanup-d1-weighted-cross-attention-goal-auxiliary-stream
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Train flow-matching DiT low-level policy on MUG_CLEANUP_D1 with GMM weighted
# cross-attention AND the dual-stream Goal Auxiliary Stream architecture.
#
# Each DiT block is now a GoalAuxiliaryStreamBlock that maintains TWO parallel
# hidden_states streams:
#   stream 1 ("main" / action stream)      — cross-attends to visual_tokens
#   stream 2 ("auxiliary" / goal stream)   — WCA to goal_tokens with GMM log-prior
# followed by joint self-attention (per-stream Q/K/V/out, one softmax over the
# concatenated sequence) and per-stream FFN. At the very end only stream 1 is
# decoded; stream 2 is discarded.
#
# Note on intent: this is NOT a hard architectural ceiling on goal influence —
# the V mixing inside joint attention lets stream-2's goal-conditioned content
# flow into stream 1 layer-by-layer. We expect goal info to still reach the
# action output, just through a more indirect channel (and via separate
# per-stream weights that may let goals organize themselves more usefully).
#
# Param cost: ~3-4× the cross-attn stack of the baseline DiT. Batch 128 should
# still fit on a single H100 80GB; if OOM, reduce dataloader.batch_size to 64.
#
# log_attention_grad_norms=true logs per-block ‖∂L/∂WCA_out‖ and
# ‖∂L/∂CA_out‖ each epoch so we can see how the dual-stream structure shifts
# the relative gradient signal between the two channels.
#
# Stages the dataset onto the compute node's local scratch ($LOCAL on PSC
# Bridges-2) before training, since reading many demo_*.h5 shards from /ocean
# is slow.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/Mug_Cleanup_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

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
DEST_DATA_DIR="${SCRATCH_ROOT}/MUG_CLEANUP_D1_Low_Level"

# --- stage dataset -------------------------------------------------------
# Parallel copy: split the top-level demo_*.h5 shards across N rsync workers via
# xargs -P. Override with RSYNC_THREADS=N env var.
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 ("vanished files") is a benign warning —
# usually rsync's own temp files (.<name>.XXXXXX) left behind by a previously
# interrupted sync. Treat it as success so it doesn't trip xargs/set -e.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Each rsync handles one top-level entry (a demo_*.h5 file). rsync stays
# resumable per-entry, so re-running the script skips already-copied files
# cheaply.
find "${SRC_DATA_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Goal-auxiliary-stream prerequisites:
#   policy.use_goal_cross_attention=true     — route GMM tokens through the dedicated
#                                              goal cross-attn module in the dual-stream
#                                              block (stream 2's WCA).
#   policy.use_weighted_cross_attention=true — use WCA (log-prior bias from GMM weights)
#                                              for stream 2's goal-side attention.
#   policy.use_goal_auxiliary_stream=true    — swap BasicTransformerBlock for
#                                              GoalAuxiliaryStreamBlock; maintain two
#                                              streams; decode only stream 1.
# Mutually exclusive with use_parallel_cross_attentions, use_gated_goal_residual,
# and use_goal_adaln_modulation (asserted at DiT init).
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/mugcleanup_D1_gmm_goal \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.use_goal_auxiliary_stream=true \
    policy.log_attention_grad_norms=true \
    policy.gmm_top_k=1024 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_Weighted_Cross_Attention_Goal_Auxiliary_Stream_MugCleanup_D1 \
    name=groot_GMM_Weighted_Cross_Attention_Goal_Auxiliary_Stream_MugCleanup_D1 \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16
