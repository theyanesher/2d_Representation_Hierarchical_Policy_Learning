#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name mug-cleanup-d1-weighted-cross-attention-goal-adaln-modulation
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Train flow-matching DiT low-level policy on MUG_CLEANUP_D1 with GMM weighted
# cross-attention AND Idea 1 (goal-AdaLN modulation). In each cross-attn block
# the WCA output is NOT added to the residual stream — it feeds GoalAdaLN as
# per-token (γ_goal, β_goal) conditioning for the visual-CA's norm. Only visual
# CA writes to the residual, so goals can only DEFORM vision's input, never
# bypass it. The goal-branch Linear inside GoalAdaLN is zero-init'd so at step
# 0 the block behaves identically to "visual-CA + FF only"; goal modulation is
# grown by the optimizer only if it actually reduces loss.
#
# Hard architectural constraint:
#   goals → MLP_goal → (γ_goal, β_goal) → shape of vision's attention input
#   goals ╳→ residual stream                                           (cannot)
#
# log_attention_grad_norms=true logs per-block ‖∂L/∂WCA_out‖ and
# ‖∂L/∂CA_out‖ to W&B each epoch so we can see whether vision actually starts
# to dominate the gradient signal under this architecture (the diagnostic we'd
# need to confirm Idea 1 is reshaping the dependence as intended).
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

# Idea-1 prerequisites:
#   policy.use_goal_cross_attention=true     — route GMM tokens through the DiT's
#                                              dedicated goal cross-attn module.
#   policy.use_weighted_cross_attention=true — log-prior bias from GMM weights.
#   policy.use_goal_adaln_modulation=true    — capture WCA output as per-token
#                                              (γ_goal, β_goal) modulation for the
#                                              visual-CA's norm; do NOT add WCA
#                                              output to the residual stream.
# Mutually exclusive with use_parallel_cross_attentions and use_gated_goal_residual
# (those modes write a goal residual; Idea 1 doesn't).
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
    policy.use_goal_adaln_modulation=true \
    policy.log_attention_grad_norms=true \
    policy.gmm_top_k=1024 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_Weighted_Cross_Attention_Goal_AdaLN_Modulation_MugCleanup_D1 \
    name=groot_GMM_Weighted_Cross_Attention_Goal_AdaLN_Modulation_MugCleanup_D1 \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16
