#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, urllib.error.URLError):
        return False


def wait_ready(process, port):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}; inspect server log")
        if healthy(port):
            return
        time.sleep(1)
    raise TimeoutError(f"server on {port} did not become ready")


def stop_port(port):
    subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False)


server = None
try:
    stop_port(1234)
    stop_port(1235)
    time.sleep(2)
    subprocess.run(["build-turbo4/bin/test-turbo4", "--gpu"], check=True)
    with open("/tmp/kv-snake-q8k-turbo4v-patched.log", "w") as log:
        server = subprocess.Popen(
            ["bash", "scripts/kv-compare/serve_mixed_context.sh"],
            env={**os.environ, "CTX": "65536", "PORT": "1235"},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        wait_ready(server, 1235)
        subprocess.run(
            ["python3", "scripts/kv-compare/speed_snake.py", "1235", "q8k-turbo4v-patched"],
            check=True, timeout=1800,
        )
finally:
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    with open("/tmp/llama-turbo4-262144.log", "w") as log:
        restored = subprocess.Popen(
            ["bash", "scripts/serve-turbo4-dflash2.sh"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    wait_ready(restored, 1234)
    print("Restored dual turbo4 server on port 1234", flush=True)
