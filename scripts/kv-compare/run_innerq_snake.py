#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
PROFILE = ROOT / "results/innerq-profile"
PROFILE.mkdir(parents=True, exist_ok=True)
ENV = {key: value for key, value in os.environ.items() if not key.startswith("TURBO_INNERQ")}


def start(mode, context, log_path):
    env = {**ENV, "CTX": str(context), mode: str(PROFILE)} if mode else {**ENV, "CTX": str(context)}
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            ["bash", "scripts/serve-turbo4-dflash2.sh"], env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(f"server exited: {process.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:1234/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return process
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    process.terminate()
    process.wait(timeout=90)
    raise TimeoutError(f"server not ready; see {log_path}")


def stop(process):
    process.terminate()
    process.wait(timeout=90)
    if process.returncode != 0:
        raise RuntimeError(f"server shutdown failed: {process.returncode}")


server = None
completed = False
try:
    server = start("TURBO_INNERQ_CALIBRATE", 65536, "/tmp/innerq-calibration.log")
    records = "\n".join(
        f"Record {i}: user={i % 17}, score={i * 13}, direction={['up', 'right', 'down', 'left'][i % 4]}. "
        "JavaScript updates the grid, checks collisions, moves the snake and generates food. "
        "使用键盘与触屏控制方向，暂停时保留状态，重新开始时清空分数。"
        for i in range(40)
    )
    payload = {
        "messages": [{"role": "user", "content": records + "\n请简述实现这个游戏需要哪些状态和函数。"}],
        "temperature": 0, "max_tokens": 128, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        calibration = json.load(response)
    (ROOT / "results/innerq-calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2))
    stop(server)
    server = None

    profiles = sorted(PROFILE.glob("*.innerq"))
    if not profiles:
        raise RuntimeError("calibration produced no profiles")
    changed = 0
    for path in profiles:
        fields = shlex.split(path.read_text())
        if len(fields) != 132 or fields[0] != "INNERQ1":
            raise RuntimeError(f"incomplete profile: {path}")
        scales = [float(value) for value in fields[4:]]
        if not all(0.5 <= value <= 2 for value in scales):
            raise RuntimeError(f"invalid profile: {path}")
        changed += sum(abs(value - 1) > 1e-6 for value in scales)
    if changed == 0:
        raise RuntimeError("all profile scales are identity; no active InnerQ test")
    print(f"Calibration: {calibration['usage']}; profiles={len(profiles)}, nonidentity_channels={changed}", flush=True)
    server = start("TURBO_INNERQ_PROFILE", 65536, "/tmp/innerq-snake.log")
    subprocess.run(
        ["python3", "scripts/kv-compare/speed_snake.py", "1234", "turbo4-innerq"],
        check=True, timeout=1800,
    )
    completed = True
finally:
    if completed:
        print(f"Keeping InnerQ server PID={server.pid}, context=65536", flush=True)
    else:
        if server is not None and server.poll() is None:
            stop(server)
        print("InnerQ validation failed; test server stopped. Inspect the logs before restarting.", flush=True)
