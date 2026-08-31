#!/usr/bin/env bash
set -euo pipefail

QWEN_RUNTIME_DIR="${QWEN_RUNTIME_DIR:-/data/theya/qwen_local}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/data/theya/models/Qwen3.6-35B-A3B-FP8}"
QWEN_SERVED_MODEL="${QWEN_SERVED_MODEL:-qwen3.6-local}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.86}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
# This workspace exports the legacy stdlib setting, but Python 3.12 removed
# distutils. vLLM's isolated environment ships setuptools' maintained copy.
export SETUPTOOLS_USE_DISTUTILS=local
QWEN_PYTHON_INCLUDE="$(find "${QWEN_RUNTIME_DIR}/python" -type f \
    -path '*/include/python3.12/Python.h' -printf '%h' -quit)"
if [[ -z "${QWEN_PYTHON_INCLUDE}" ]]; then
    echo "Python 3.12 C headers are missing; run scripts/setup_qwen_local.sh" >&2
    exit 1
fi
export CPATH="${QWEN_PYTHON_INCLUDE}${CPATH:+:${CPATH}}"
# FlashInfer JIT-compiles kernels in a subprocess and looks up ninja on PATH.
# The venv is never activated (we exec its vllm directly), so add its bin dir.
export PATH="${QWEN_RUNTIME_DIR}/.venv/bin:${PATH}"

VLLM_BIN="${QWEN_RUNTIME_DIR}/.venv/bin/vllm"
if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "Local Qwen runtime is missing; run scripts/setup_qwen_local.sh" >&2
    exit 1
fi
if [[ ! -f "${QWEN_MODEL_DIR}/config.json" ]]; then
    echo "Local Qwen checkpoint is missing; run scripts/setup_qwen_local.sh" >&2
    exit 1
fi

exec "${VLLM_BIN}" serve "${QWEN_MODEL_DIR}" \
    --served-model-name "${QWEN_SERVED_MODEL}" \
    --host "${QWEN_HOST}" \
    --port "${QWEN_PORT}" \
    --tensor-parallel-size 2 \
    --max-model-len "${QWEN_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 16}'
