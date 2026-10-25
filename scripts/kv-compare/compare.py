#!/usr/bin/env python3
import json
import os
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: compare.py <q8.jsonl> <turbo4.jsonl>")


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def good(row):
    if row.get("error"):
        return False
    answer = "".join((row.get("answer") or "").split()).lower()
    return all("".join(str(x).split()).lower() in answer for x in row.get("expect", []))

q8 = load(sys.argv[1])
t4 = load(sys.argv[2])
ids = list(q8)
rows = []
for pid in ids:
    a, b = q8[pid], t4.get(pid, {})
    rows.append((pid, a, b, good(a), good(b)))

os.makedirs("results", exist_ok=True)
with open("results/kv-compare.md", "w", encoding="utf-8") as out:
    out.write("# turbo4_0 vs q8_0 KV 对照结果\n\n")
    out.write("| ID | 类型 | q8_0 | turbo4_0 | q8 秒 | turbo4 秒 | q8 prompt tok | turbo4 prompt tok |\n")
    out.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for pid, a, b, oka, okb in rows:
        out.write("| {} | {} | {} | {} | {:.1f} | {:.1f} | {} | {} |\n".format(
            pid, a.get("kind", ""), "PASS" if oka else "FAIL", "PASS" if okb else "FAIL",
            a.get("elapsed_sec", 0), b.get("elapsed_sec", 0),
            a.get("prompt_tokens", "-"), b.get("prompt_tokens", "-")))
    out.write("\n## 汇总\n\n")
    for kind in ("t0", "t1", "t2"):
        subset = [(a, b, oka, okb) for _, a, b, oka, okb in rows if a.get("kind") == kind]
        if subset:
            out.write("- {}: q8_0 {}/{}，turbo4_0 {}/{}\n".format(
                kind, sum(x[2] for x in subset), len(subset), sum(x[3] for x in subset), len(subset)))
    out.write("\n## 失败项目答案\n\n")
    for pid, a, b, oka, okb in rows:
        if not oka or not okb:
            out.write("### {}\n\n".format(pid))
            out.write("- 期望：`{}`\n".format("`; `".join(a.get("expect", []))))
            out.write("- q8_0：`{}`\n".format((a.get("answer") or "").replace("\n", " ")[:1000]))
            out.write("- turbo4_0：`{}`\n\n".format((b.get("answer") or "").replace("\n", " ")[:1000]))

for kind in ("t0", "t1", "t2"):
    subset = [(a, b, oka, okb) for _, a, b, oka, okb in rows if a.get("kind") == kind]
    if subset:
        print("{}: q8_0 {}/{} turbo4_0 {}/{}".format(
            kind, sum(x[2] for x in subset), len(subset), sum(x[3] for x in subset), len(subset)))
print("report: results/kv-compare.md")
