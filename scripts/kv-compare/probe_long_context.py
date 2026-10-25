#!/usr/bin/env python3
import json
import os
import time
import urllib.request

BASE = "http://127.0.0.1:1234"
PHRASE = "The quick brown fox jumps over the lazy dog. "
TARGET = int(os.environ.get("TARGET_TOKENS", "190000"))


def post(path, payload, timeout):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())

sample = post("/tokenize", {"content": PHRASE * 1000}, 60)
rate = len(sample["tokens"]) / 1000
repeats = max(1, int(TARGET / rate))
content = PHRASE * repeats
tokens = post("/tokenize", {"content": content}, 180)["tokens"]
print("content_tokens:", len(tokens), "chars:", len(content), flush=True)

payload = {
    "model": "long-context-probe",
    "messages": [{
        "role": "user",
        "content": content + "\n\n请基于上面的长上下文，输出一份详细的贪吃蛇游戏设计说明，包含玩法、控制方式和实现要点。请直接回答，不要只回复确认词。",
    }],
    "temperature": 0,
    "max_tokens": 256,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
started = time.monotonic()
result = post("/v1/chat/completions", payload, 1800)
print("elapsed_seconds:", round(time.monotonic() - started, 3))
print(json.dumps({
    "usage": result.get("usage"),
    "timings": result.get("timings"),
    "answer": ((result.get("choices") or [{}])[0].get("message") or {}).get("content", ""),
}, ensure_ascii=False, indent=2))
