#!/bin/bash
#SBATCH -N 1                  # Number of nodes
#SBATCH --ntasks-per-node=1   # single python process (one rollout at a time)
#SBATCH --cpus-per-task=12    # MuJoCo stepping + rendering + ffmpeg encode
#SBATCH -p ROBO
#SBATCH --gpus=h100:1         # needed for the DiT forward AND for EGL rendering
#SBATCH -t 10:00:00           # 3 seeds x 50 episodes x 800 steps, sequential
#SBATCH --job-name a2-eval-coffee-c1-0.1
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Mimicgen_Inference/2d_Representation_Hierarchical_Policy_Learning/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Mimicgen_Inference/2d_Representation_Hierarchical_Policy_Learning/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 eval on Coffee_Preperation_D1 (c1 = 0.1) — all three seeds.
# ===========================================================================
# This is a THIN WRAPPER around the interactive script in this same folder:
#
#   eval_approach2_2d_dit_low_level_coffee_preperation_d1_DINOV2_c1_0.1_ALL_SEEDS.sh
#
# All eval logic — the 3 seeds, the per-seed output tree, and the resume+merge
# bookkeeping — lives there and is NOT duplicated here. This file only supplies
# the SLURM allocation and the environment that a login-node shell gives for
# free (GPU-backed EGL, node-local scratch). Keeping it a wrapper means the
# batch and interactive paths can never drift apart.
#
# Submit:
#   sbatch sbatch_eval_approach2_coffee_preperation_d1_c1_0.1_ALL_SEEDS.sh
#
# The eval is RESUMABLE. If the 48h budget runs out mid-run, just resubmit —
# the inner script counts completed episodes in results.jsonl and continues.
# ===========================================================================

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_ROOT="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Mimicgen_Inference/2d_Representation_Hierarchical_Policy_Learning"

INNER_SCRIPT="${SCRIPT_DIR}/eval_approach2_2d_dit_low_level_coffee_preperation_d1_DINOV2_c1_0.1_ALL_SEEDS.sh"

mkdir -p "${INFERENCE_ROOT}/logs"

if [[ ! -x "${INNER_SCRIPT}" ]]; then
    echo "[ERROR] inner eval script not found or not executable: ${INNER_SCRIPT}" >&2
    exit 1
fi

# --- headless rendering --------------------------------------------------
# robosuite renders through EGL on the allocated GPU. DISPLAY is set only so
# that anything probing it finds a value; nothing actually connects to an X
# server. Without MUJOCO_GL/PYOPENGL_PLATFORM the job dies on a compute node.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISPLAY="${DISPLAY:-:99}"

# --- keep scratch off $HOME ----------------------------------------------
# $HOME is quota-tight (25 GB) and both MuJoCo and the ffmpeg/PyAV encoder
# write temp files. Node-local scratch is faster and cannot exhaust the quota.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export TMPDIR="/local/slurm-${SLURM_JOB_ID}/local/a2_eval_coffee"
else
    export TMPDIR="${TMPDIR:-/tmp}/a2_eval_coffee_$$"
fi
mkdir -p "${TMPDIR}"

# --- interpreter isolation -----------------------------------------------
# PYTHONNOUSERSITE stops ~/.local/lib packages from shadowing the pixi env.
# The inner script invokes ${INFERENCE_ROOT}/.pixi/envs/default/bin/python
# directly, so no pixi activation is required here.
export PYTHONNOUSERSITE=1
export PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache
export PATH="$HOME/.pixi/bin:$PATH"

echo "[job]   ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo "[gpu]   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[tmp]   ${TMPDIR}"
echo "[inner] ${INNER_SCRIPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${INFERENCE_ROOT}"
exec "${INNER_SCRIPT}"
