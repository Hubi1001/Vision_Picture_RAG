#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
set -a; source "${ROOT_DIR}/.env"; set +a

OUT_HOST="${ROOT_DIR}/${OUTPUT_DIR}"
MODEL_PATH="${OUT_HOST}/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"

docker run \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${MODEL_PATH}:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "${TLLM_IMAGE}" \
  bash -c '
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Paris is great because" \
      --max_tokens 64
  '
