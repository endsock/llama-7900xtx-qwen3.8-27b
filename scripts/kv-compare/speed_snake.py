#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

if len(sys.argv) != 3:
    raise SystemExit("usage: speed_snake.py <port> <name>")

port = int(sys.argv[1])
name = sys.argv[2]
base = "http://127.0.0.1:{}".format(port)
prompt = """请输出一个完整、可直接运行的贪吃蛇游戏。
要求：使用单个 HTML 文件，内含 HTML、CSS 和原生 JavaScript；支持键盘方向键、开始/暂停、重新开始、分数、最高分、碰撞检测、食物随机生成和移动端按钮；界面要完整美观。请只输出最终代码，不要解释，不要省略任何代码。代码放在一个 html 代码块中。"""
payload = {
    "model": "snake-speed",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 42,
    "max_tokens": 65535,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

for _ in range(120):
    try:
        req = urllib.request.Request(base + "/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            if json.loads(response.read()).get("status") == "ok":
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("server is not healthy: {}".format(base))

started = time.monotonic()
request = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.loads(response.read().decode())
elapsed = time.monotonic() - started

usage = result.get("usage", {})
timings = result.get("timings", {})
answer = ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
summary = {
    "name": name,
    "max_tokens": 65535,
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "total_tokens": usage.get("total_tokens"),
    "wall_seconds": elapsed,
    "wall_tokens_per_second": (usage.get("completion_tokens", 0) / elapsed) if elapsed else 0,
    "server_timings": timings,
    "answer_chars": len(answer),
}
with open("results/snake-{}.json".format(name), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "answer": answer}, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
