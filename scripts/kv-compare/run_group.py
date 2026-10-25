#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request

if len(sys.argv) != 4:
    raise SystemExit("usage: run_group.py <port> <probes.json> <results.json>")

port, probes_path, results_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
base = "http://127.0.0.1:{}".format(port)


def request(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


for _ in range(180):
    try:
        if request("/health", timeout=5).get("status") == "ok":
            break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("server did not become healthy: {}".format(base))

with open(probes_path, encoding="utf-8") as f:
    probes = json.load(f)

os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
with open(results_path, "w", encoding="utf-8") as out:
    for i, probe in enumerate(probes, 1):
        payload = {
            "model": "kv-compare",
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 42,
            "max_tokens": 256,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        error = None
        response = None
        try:
            response = request("/v1/chat/completions", payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception) as exc:
            error = str(exc)
        elapsed = time.monotonic() - started
        choice = ((response or {}).get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = (response or {}).get("usage") or {}
        result = {
            "id": probe["id"],
            "kind": probe["kind"],
            "expect": probe["expect"],
            "target_tokens": probe.get("target_tokens"),
            "answer": message.get("content", "") if error is None else "",
            "error": error,
            "elapsed_sec": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()
        print("[{}/{}] {} {:.1f}s {}".format(
            i, len(probes), probe["id"], elapsed, "ERROR: " + error if error else "ok"), flush=True)
