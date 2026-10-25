#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAMA_ATTN_ROT_DISABLE=1

exec "$ROOT/build-turbo4/bin/llama-server" \
  -m /data/project/ai/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --model-draft /data/project/ai/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type ngram-map-k4v,draft-dflash \
  --spec-draft-n-max 5 \
  --spec-draft-p-min 0.4 \
  --spec-ngram-map-k4v-size-n 12 \
  --spec-ngram-map-k4v-size-m 48 \
  --spec-ngram-map-k4v-min-hits 1 \
  -ngld 99 \
  -ctk turbo4_0 -ctv turbo4_0 \
  -ngl 99 -c 262144 \
  -b 2048 -ub 256 \
  -fa on --jinja \
  --reasoning-effort medium \
  --no-kv-unified --parallel 1 \
  --ctx-checkpoints 2 --cache-ram 4096 \
  --no-warmup --no-mmproj \
  --host 0.0.0.0 --port 1234
