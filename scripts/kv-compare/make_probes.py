#!/usr/bin/env python3
# Generate deterministic KV-compare probes: T0 short baseline, T1 needle-in-haystack, T2 long-table QA.
# Usage: python3 make_probes.py [out.json]
import json
import random
import sys

SEED = 42
OUT = sys.argv[1] if len(sys.argv) > 1 else "probes.json"

rng = random.Random(SEED)

# ---- T0: short-context sanity questions ----
T0 = [
    ("t0-1-mul",  "请直接计算并只回答数字：17 * 23 = ?", ["391"]),
    ("t0-2-perim", "一个正方形边长 12 厘米，它的周长是多少厘米？只回答数字和单位。", ["48"]),
    ("t0-3-moon", "地球有几颗天然卫星？只回答数字。", ["1"]),
    ("t0-4-dup",  "在词语序列『苹果、香蕉、苹果、梨』中，重复出现的是哪个水果？只回答水果名。", ["苹果"]),
    ("t0-5-age",  "小明比小红大 3 岁，小红 8 岁，小明几岁？只回答数字。", ["11"]),
]

# ---- T1: needle-in-haystack ----
# target prompt token counts (approx, capped to leave output room under -c 65536)
LENGTHS = [2000, 8000, 16000, 32000, 58000]
DEPTHS = [0.10, 0.50, 0.90]
# rough chars-per-token for this Chinese-heavy corpus; runner reports actual usage
CHARS_PER_TOKEN = 1.42

FILLER_TPL = [
    "第{a}号观测站在{m}月的例行记录显示，设备读数稳定在{b}与{c}之间，维护人员确认无需干预。",
    "关于{a}区的风场数据，本周平均值为{b}，较上一周期略有波动，属于正常范围，未见异常告警。",
    "巡检日志记载，{m}月{d}日对{a}号机组进行了例行检查，各指标均落在{b}附近，运行平稳。",
    "档案室补充了一份{m}季度的说明文件，编号为{a}，内容涉及{b}项参数的复核结论，无异常。",
    "值班记录第{a}条指出，{m}月{d}日凌晨的读数{b}在预期区间内，值班员已按规程归档处理。",
]
MONTHS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]


def make_filler(n_chars):
    parts = []
    total = 0
    i = 0
    while total < n_chars:
        s = FILLER_TPL[i % len(FILLER_TPL)].format(
            a=rng.randint(100, 999), b=rng.randint(10, 99),
            c=rng.randint(100, 999), m=rng.choice(MONTHS), d=rng.randint(1, 28))
        parts.append(s)
        total += len(s)
        i += 1
    return "".join(parts)[:n_chars]


def make_needle(idx):
    return "TQ-{a}-{b}-{c}".format(
        a=rng.randint(1000, 9999),
        b=rng.choice(["紫罗兰灯塔", "青苔铜钟", "灰白鲸群", "琥珀灯塔", "松木风车", "银线河湾"]),
        c=rng.choice(["MAV", "KRN", "PLT", "QWB", "ZHX"]))


needles = {}
probes = []
for tid, prompt, keys in T0:
    probes.append({"id": tid, "kind": "t0", "prompt": prompt, "expect": keys,
                   "target_tokens": None})

for li, tgt in enumerate(LENGTHS):
    n_chars = int(tgt * CHARS_PER_TOKEN)
    filler = make_filler(n_chars)
    for di, depth in enumerate(DEPTHS):
        pid = "t1-l{}/d{}".format(li, di)
        needle = make_needle(len(needles))
        needles[pid] = needle
        n_seg = len(filler) // 100
        cut = int(n_seg * 100 * depth)
        doc = filler[:cut] + "\n" + needle + "\n" + filler[cut:]
        prompt = doc + "\n\n问题：文档中隐藏了一行以 TQ- 开头的暗语代码，请完整复述这一行，不要包含其他内容。"
        probes.append({"id": pid, "kind": "t1", "prompt": prompt, "expect": [needle],
                       "target_tokens": tgt})

# ---- T2: long structured table, multi-fact QA ----
rng2 = random.Random(SEED + 1)
DEPTS = ["物流", "研发", "市场", "财务", "客服"]
CITIES = ["成都", "武汉", "西安", "杭州", "南京"]
n_records = 400
records = []
for i in range(1, n_records + 1):
    records.append("EMP-{a} | 部门: {d} | 城市: {c} | 工资: {w} 元 | 入职: {y}-{m}-{dd}".format(
        a="{:04d}".format(i), d=rng2.choice(DEPTS), c=rng2.choice(CITIES),
        w=rng2.randint(6000, 40000), y=rng2.randint(2005, 2023),
        m="{:02d}".format(rng2.randint(1, 12)), dd="{:02d}".format(rng2.randint(1, 28))))
table = "\n".join(records)
# pad to ~40k tokens with filler so the table sits inside a long document
pad = make_filler(int(36000 * CHARS_PER_TOKEN))
doc2 = pad + "\n\n以下是员工花名册，共 {} 条记录：\n".format(n_records) + table

best = max(records, key=lambda r: int(r.split("工资: ")[1].split()[0]))
best_id = best.split(" | ")[0]
best_w = best.split("工资: ")[1].split()[0]
dept_cnt = sum(1 for r in records if "部门: 物流" in r)
cd = [r for r in records if "城市: 成都" in r]
cd_id = min(cd, key=lambda r: r.split("入职: ")[1]) .split(" | ")[0]
e42 = [r for r in records if r.startswith("EMP-0042")][0]
e42_city = e42.split("城市: ")[1].split(" |")[0]

T2 = [
    ("t2-1-max", doc2 + "\n\n问题：花名册中工资最高的员工，其编号和工资各是多少？只回答：编号，工资。", [best_id, best_w]),
    ("t2-2-count", doc2 + "\n\n问题：花名册中部门为『物流』的员工共有多少人？只回答数字。", [str(dept_cnt)]),
    ("t2-3-earliest", doc2 + "\n\n问题：花名册中城市为『成都』且入职日期最早的员工编号是多少？只回答编号。", [cd_id]),
    ("t2-4-lookup", doc2 + "\n\n问题：EMP-0042 在哪座城市工作？只回答城市名。", [e42_city]),
]
for tid, prompt, keys in T2:
    probes.append({"id": tid, "kind": "t2", "prompt": prompt, "expect": keys,
                   "target_tokens": 40000})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(probes, f, ensure_ascii=False, indent=1)
print("wrote {} probes -> {}".format(len(probes), OUT))
print("t2 answers:", {"max": [best_id, best_w], "logistics": dept_cnt,
                      "chengdu_earliest": cd_id, "emp42_city": e42_city})
