#!/usr/bin/env python3
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
result = json.loads((ROOT / "results/snake-turbo4-innerq.json").read_text())
answer = result["answer"]
match = re.fullmatch(r"\s*```html\s*\n(.*?)\n```\s*", answer, re.DOTALL)
if not match:
    raise RuntimeError("response is not a complete fenced HTML document")
html = match.group(1)
if "</html>" not in html.lower() or "</body>" not in html.lower():
    raise RuntimeError("HTML document is incomplete")
output = ROOT / "results/snake-turbo4-innerq.html"
output.write_text(html)


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.inside = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.inside = False
            self.parts.append("\n")

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


parser = Scripts()
parser.feed(html)
if not parser.parts:
    raise RuntimeError("HTML has no inline JavaScript")
js = Path("/tmp/snake-turbo4-innerq.js")
js.write_text("".join(parser.parts))
subprocess.run(["node", "--check", str(js)], check=True)
with urllib.request.urlopen("http://127.0.0.1:1234/props", timeout=10) as response:
    props = json.load(response)
with urllib.request.urlopen("http://127.0.0.1:1234/health", timeout=10) as response:
    health = json.load(response)
print(json.dumps({
    "html": str(output), "javascript_syntax": "PASS", "health": health,
    "context": props.get("default_generation_settings", {}).get("n_ctx"),
    "slots": props.get("total_slots"),
    "note": "Syntax and document completeness only; browser gameplay not tested.",
}, ensure_ascii=False, indent=2))
