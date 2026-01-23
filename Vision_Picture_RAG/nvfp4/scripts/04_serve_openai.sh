#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
set -a; source "${ROOT_DIR}/.env"; set +a

OUT_HOST="${ROOT_DIR}/${OUTPUT_DIR}"
MODEL_PATH="${OUT_HOST}/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"

docker run \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${MODEL_PATH}:/workspace/model" \
  -d --rm --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "${TLLM_IMAGE}" \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --port "${PORT}"
