#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CTK="${CTK:?set CTK}"
CTV="${CTV:?set CTV}"
PORT="${PORT:?set PORT}"

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
  -ctk "$CTK" -ctv "$CTV" \
  -ngl 99 -c 65536 \
  -b 2048 -ub 512 \
  -fa on --jinja \
  --no-kv-unified --parallel 1 \
  --ctx-checkpoints 2 --cache-ram 4096 \
  --no-warmup --no-mmproj \
  --host 127.0.0.1 --port "$PORT"
