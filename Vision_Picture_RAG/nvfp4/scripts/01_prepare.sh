#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
set -a; source "${ROOT_DIR}/.env"; set +a

mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}"
chmod 755 "${ROOT_DIR}/${OUTPUT_DIR}"
echo "OK: ${ROOT_DIR}/${OUTPUT_DIR} gotowe"
