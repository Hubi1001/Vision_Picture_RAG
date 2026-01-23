#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
set -a; source "${ROOT_DIR}/.env"; set +a

OUT_HOST="${ROOT_DIR}/${OUTPUT_DIR}"
mkdir -p "${OUT_HOST}"
chmod 755 "${OUT_HOST}"

docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${OUT_HOST}:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_ID="${MODEL_ID}" \
  -e MODEL_OPTIMIZER_GIT_TAG="${MODEL_OPTIMIZER_GIT_TAG}" \
  -e TP="${TP}" \
  "${TLLM_IMAGE}" \
  bash -c '
    git clone --single-branch --branch ${MODEL_OPTIMIZER_GIT_TAG} https://github.com/NVIDIA/Model-Optimizer && \
    cd Model-Optimizer && pip install -e .[dev] && \
    export ROOT_SAVE_PATH="/workspace/output_models" && \
    bash examples/llm_ptq/scripts/huggingface_example.sh \
      --model "${MODEL_ID}" \
      --quant nvfp4 \
      --tp ${TP} \
      --export_fmt hf
  '
