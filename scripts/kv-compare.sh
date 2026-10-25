#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/scripts/kv-compare"
RESULTS="$DIR/results"
PORT=1235

mkdir -p "$RESULTS"
python3 "$DIR/make_probes.py" "$DIR/probes.json"

kill_port() {
    local port="$1"
    local pids
    pids="$(pgrep -f "build-turbo4/bin/llama-server.*--port ${port}" || true)"
    if [[ -n "$pids" ]]; then
        kill $pids || true
        for _ in $(seq 1 60); do
            pgrep -f "build-turbo4/bin/llama-server.*--port ${port}" >/dev/null || return 0
            sleep 1
        done
        kill -9 $pids || true
    fi
}

start_group() {
    local name="$1" ctk="$2" ctv="$3"
    CTK="$ctk" CTV="$ctv" PORT="$PORT" \
        nohup bash "$DIR/serve_group.sh" >"/tmp/kv-compare-${name}.log" 2>&1 &
    echo $! >"/tmp/kv-compare-${name}.pid"
    for _ in $(seq 1 300); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$(cat "/tmp/kv-compare-${name}.pid")" 2>/dev/null; then
            echo "${name} server exited; see /tmp/kv-compare-${name}.log" >&2
            return 1
        fi
        sleep 2
    done
    echo "${name} server did not become healthy" >&2
    return 1
}

run_group() {
    local name="$1"
    python3 "$DIR/run_group.py" "$PORT" "$DIR/probes.json" "$RESULTS/${name}.jsonl"
    kill_port "$PORT"
}

restore() {
    kill_port "$PORT"
    if [[ "${ORIGINAL_RUNNING:-0}" == 1 ]]; then
        nohup bash "$ROOT/scripts/serve-turbo4-dflash2.sh" >/tmp/llama-turbo4-262144.log 2>&1 &
        echo "restoring original turbo4 service on port 1234 (pid $!)"
    fi
}
trap restore EXIT

if pgrep -f 'build-turbo4/bin/llama-server.*--port 1234' >/dev/null; then
    ORIGINAL_RUNNING=1
else
    ORIGINAL_RUNNING=0
fi
kill_port 1234

start_group q8 q8_0 q8_0
run_group q8

start_group turbo4 turbo4_0 turbo4_0
run_group turbo4

python3 "$DIR/compare.py" "$RESULTS/q8.jsonl" "$RESULTS/turbo4.jsonl"
