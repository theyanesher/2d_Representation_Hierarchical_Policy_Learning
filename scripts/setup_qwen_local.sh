#!/usr/bin/env bash
set -euo pipefail

# Keep the rapidly moving Qwen/vLLM stack isolated from the robot-policy env.
QWEN_RUNTIME_DIR="${QWEN_RUNTIME_DIR:-/data/theya/qwen_local}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/data/theya/models/Qwen3.6-35B-A3B-FP8}"
QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"

command -v uv >/dev/null || {
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

mkdir -p "${QWEN_RUNTIME_DIR}" "${QWEN_MODEL_DIR}"
uv python install 3.12 --install-dir "${QWEN_RUNTIME_DIR}/python" --no-bin
QWEN_PYTHON="$(find "${QWEN_RUNTIME_DIR}/python" -type f \
    -path '*/bin/python3.12' -print -quit)"
if [[ -z "${QWEN_PYTHON}" ]]; then
    echo "uv-managed Python 3.12 was installed but could not be located" >&2
    exit 1
fi
if [[ ! -x "${QWEN_RUNTIME_DIR}/.venv/bin/python" ]]; then
    uv venv --python "${QWEN_PYTHON}" "${QWEN_RUNTIME_DIR}/.venv"
fi
uv pip install --python "${QWEN_RUNTIME_DIR}/.venv/bin/python" \
    'vllm==0.27.1' huggingface_hub --torch-backend=auto
# Direct HTTP is more reliable than Xet for this large, many-shard checkpoint
# on the target machine and still resumes incomplete files.
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
    "${QWEN_RUNTIME_DIR}/.venv/bin/hf" download "${QWEN_MODEL_REPO}" \
    --local-dir "${QWEN_MODEL_DIR}"

echo "Local Qwen setup complete: ${QWEN_MODEL_DIR}"
echo "Start it with: scripts/serve_qwen_local.sh"
