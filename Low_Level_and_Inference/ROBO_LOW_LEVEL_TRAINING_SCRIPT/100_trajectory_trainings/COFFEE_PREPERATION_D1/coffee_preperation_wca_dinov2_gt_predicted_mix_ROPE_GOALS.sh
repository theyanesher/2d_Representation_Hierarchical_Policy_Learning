#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name coffee-prep-d1-wca-ropegoals-100demo-dinov2-gtmix
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo GT/predicted goal-MIX run on COFFEE_PREPERATION_D1, with 3D-GROUNDED
# VISION (RoPE4D trunk after DINOv2).
#
# Pair-compare against coffee_preperation_wca_dinov2_gt_predicted_mix.sh, which
# is this run without the trunk. Exactly two things differ:
#     task           ..._gmm_goal_gt_mix  ->  ..._gmm_goal_gt_mix_rope
#     visual_encoder dinov2               ->  dinov2_rope4d_grounded
# Everything else — gt_mix_p schedule, gmm_top_k=6, WCA, batch size, epochs,
# seed — is unchanged.
#
# What the encoder swap does: each DINOv2 patch token gets a world-frame 3D
# anchor (depth unprojected with the per-timestep camera-to-world extrinsics),
# the 4 present_gripper_pts keypoints join as extra grounded tokens, and 2 layers
# of 4D-RoPE self-attention fuse them so attention depends on real metric offsets
# (dx, dy, dz, dt). The result replaces the raw DINOv2 tokens as the DiT's
# encoder_hidden_states.
#
# ADDITIONALLY, the top-6 GMM GOAL CANDIDATES join the trunk as grounded tokens:
#
#   trunk = [ 1024 patches ; 8 gripper keypoints ; 2 steps x 6 goals ] = 1044
#
# Each goal token sits at goal_gripper_pts[..., 3, :] — keypoint 3, which is
# bit-identically obs/state[:3], the EE frame origin. (The 4-point centroid would
# have been a systematic ~29.5 mm off that convention, comparable to the entire
# 25-35 mm nearest-anchor scale.) The content still carries all 12 coordinates.
#
# The mixture weights enter as an attention prior on the goal KEYS:
#     logits = alpha_trunk*(q.k)/sqrt(d) + beta_trunk*log(pi/pi_max)
#                                        + gamma_trunk*log(pi_max)
# The pi_max normalisation matters here in a way it never did for WCA: WCA's
# softmax spans goal candidates ONLY, so subtracting a constant is a no-op by
# shift invariance. The trunk's softmax also spans 1032 patch keys, so a raw
# log(pi) would push goals DOWN against patches by an amount set by how peaked
# the mixture happens to be — quieting the goal stream exactly when the high-level
# is uncertain. gamma_trunk makes that its own knob rather than a side effect of
# beta. alpha_trunk is applied by scaling the goal KEYS (which scales their
# column of the score matrix); scaling q instead would rescale every logit.
#
# The GOAL PATHWAY IS OTHERWISE UNTOUCHED: WCA still runs over the same top-6
# candidates, so action tokens keep their direct weighted access. The trunk is an
# ADDITIONAL geometric route, not a replacement. Goal and gripper tokens are
# context only — discarded at the trunk output; only the 1024 patch tokens reach
# the DiT.
#
# Pair-compare against coffee_preperation_wca_dinov2_gt_predicted_mix_ROPE.sh:
# same task, same everything, differing only in whether goals are in the trunk.
#
# CONFOUND to keep in mind: the trunk adds ~25M parameters (2 attention layers)
# on top of the 3D grounding, so a win over the baseline is "grounding AND
# capacity", not grounding alone. The control that isolates geometry would be the
# same trunk with plain self-attention instead of RoPE4D.
#
# The task config additionally requests cam{0,1}_depth / _intrinsic / _extrinsic
# and present_gripper_pts (all already in the LOW_LEVEL_WITH_GMM h5 files) and
# sets identity_normalize_depth=true so depth reaches the encoder in METRES.
#
# Trains from scratch — the architecture differs from the non-RoPE baseline, so
# those checkpoints will not load.
#
# gt_mix_p lives in the config (0.5); override here or at launch:
#   GT_MIX_P=0.3 sbatch this_script.sh
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- gt-mix probability (lives in the config; overridable here) -----------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth goals per sample)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/COFFEE_PREPERATION_D1"
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

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_gmm_goal_gt_mix_rope \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    visual_encoder=dinov2_rope4d_grounded_goals \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_ROPE_GOALS_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_GTMIX_p${GT_MIX_P} \
    name=groot_GMM_WCA_ROPE_GOALS_${NUM_DEMOS}demo_dinov2_Coffee_Preperation_D1_GTMIX_p${GT_MIX_P} \
    training.checkpoint_every=5 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16
