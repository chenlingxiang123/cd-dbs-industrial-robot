# -*- coding: utf-8 -*-
"""Summarise CD-DBS run into paper table + SVG figures (no matplotlib).

Usage: python make_report.py <report.json> <samples.csv> <out_prefix>
Figures are hand-built SVG so they never depend on C-extension plotting libs.
"""
import csv
import json
import sys


def fmt(x):
    return f"{100*float(x):.2f}%"


METRIC_KEYS = [
    ("parse_fail_rate", "解析失败率"),
    ("danger_rate", "危险指令率↓"),
    ("viol_rate", "约束违反率"),
    ("semantic_wrong_rate", "语义错误率"),
    ("safe_exec_rate", "安全执行率↑"),
    ("degrade_rate", "拒发降级率"),
    ("hit_rate", "自检命中率"),
    ("false_alarm_rate", "误拦率"),
]
METHODS = ["baseline", "constraint_decoding",
           "dual_branch_selfcheck", "cd_plus_dbs"]
ZH = {"baseline": "Baseline", "constraint_decoding": "+CD 约束解码",
      "dual_branch_selfcheck": "+DBS 双分支自检",
      "cd_plus_dbs": "+CD+DBS 完整方法"}
COLOR = {"baseline": "#8a8a8a", "constraint_decoding": "#4c72b0",
         "dual_branch_selfcheck": "#dd8452", "cd_plus_dbs": "#55a868"}


def build_table(rep):
    lines = ["| 方法 | " + " | ".join(n for _, n in METRIC_KEYS) +
             " | 延迟(ms) |", "|---|" + "---|" * (len(METRIC_KEYS) + 1)]
    for m in METHODS:
        d = rep[m]
        cells = [fmt(d[k]) if k in d else "—" for k, _ in METRIC_KEYS]
        lines.append("| " + ZH[m] + " | " + " | ".join(cells) +
                     f" | {d['latency_ms']:.0f} |")
    ds = rep.get("danger_subset")
    if ds:
        lines += ["", "## 危险工单子集（n=%d）" % ds["n_danger"], "",
                  "| 指标 | 比率 | 95%CI |", "|---|---|---|"]
        namemap = {"baseline_unsafe_exec": "Baseline 危险执行率",
                   "baseline_copy_viol": "Baseline 照抄违规值率",
                   "cd_field_corrected": "CD 冲突字段修正率",
                   "cd_all_safe": "CD 全约束合法率"}
        for k, name in namemap.items():
            v = ds[k]
            lo, hi = v["ci95"]
            lines.append(f"| {name} | {fmt(v['rate'])} | [{lo:.2%}, {hi:.2%}] |")
    return "\n".join(lines)


def grouped_bar_svg(rep, path):
    """4 metrics x 4 methods grouped bar chart, pure SVG."""
    keys = ["danger_rate", "viol_rate", "safe_exec_rate", "semantic_wrong_rate"]
    names = ["危险指令率", "约束违反率", "安全执行率", "语义错误率"]
    W, H = 900, 480
    padL, padB, padT = 70, 90, 50
    plotW, plotH = W - padL - 30, H - padT - padB
    gw = plotW / len(keys)
    bw = gw * 0.72 / len(METHODS)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="24" text-anchor="middle" font-size="17" font-weight="bold">CD-DBS 四组方法核心指标 (n={rep["n"]})</text>']
    # y grid 0..100
    for g in range(0, 101, 20):
        y = padT + plotH * (1 - g / 100)
        parts.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{W-30}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{padL-8}" y="{y+4:.1f}" text-anchor="end" font-size="12">{g}%</text>')
    for i, k in enumerate(keys):
        gx = padL + i * gw
        for j, m in enumerate(METHODS):
            v = 100 * float(rep[m].get(k, 0.0))
            x = gx + gw * 0.14 + j * bw
            h = plotH * v / 100
            y = padT + plotH - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" height="{h:.1f}" fill="{COLOR[m]}"/>')
            parts.append(f'<text x="{x+bw*0.45:.1f}" y="{y-3:.1f}" text-anchor="middle" font-size="10">{v:.0f}</text>')
        cx = gx + gw / 2
        parts.append(f'<text x="{cx:.1f}" y="{padT+plotH+22}" text-anchor="middle" font-size="14">{names[i]}</text>')
    # legend
    lx = padL
    ly = H - 24
    for m in METHODS:
        parts.append(f'<rect x="{lx}" y="{ly-11}" width="14" height="14" fill="{COLOR[m]}"/>')
        parts.append(f'<text x="{lx+18}" y="{ly}" font-size="12">{ZH[m]}</text>')
        lx += 210
    parts.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(parts))


def latency_svg(rep, path):
    W, H = 620, 400
    padL, padB, padT = 60, 90, 50
    plotW, plotH = W - padL - 30, H - padT - padB
    lats = [rep[m]["latency_ms"] for m in METHODS]
    vmax = max(lats) * 1.15
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">平均推理延迟 (ms/条, 真实计时)</text>']
    bw = plotW / len(METHODS) * 0.6
    for i, (m, v) in enumerate(zip(METHODS, lats)):
        x = padL + (i + 0.5) * (plotW / len(METHODS)) - bw / 2
        h = plotH * v / vmax
        y = padT + plotH - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{COLOR[m]}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="12">{v:.0f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{padT+plotH+18}" text-anchor="middle" font-size="11">{ZH[m]}</text>')
    parts.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(parts))


def danger_subset_svg(ds, path):
    W, H = 640, 400
    padL, padB, padT = 55, 95, 45
    plotW, plotH = W - padL - 25, H - padT - padB
    cats = [("baseline_unsafe_exec", "危险执行率", "#c0392b"),
            ("baseline_copy_viol", "照抄违规值", "#c0392b"),
            ("cd_field_corrected", "CD字段修正", "#27ae60"),
            ("cd_all_safe", "CD全约束合法", "#27ae60")]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="22" text-anchor="middle" font-size="16" font-weight="bold">危险工单子集 (n={ds["n_danger"]})</text>']
    for g in range(0, 101, 20):
        y = padT + plotH * (1 - g / 100)
        parts.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{W-25}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{padL-6}" y="{y+4:.1f}" text-anchor="end" font-size="11">{g}%</text>')
    bw = plotW / len(cats) * 0.55
    for i, (k, name, col) in enumerate(cats):
        v = 100 * float(ds[k]["rate"])
        x = padL + (i + 0.5) * (plotW / len(cats)) - bw / 2
        h = plotH * v / 100
        y = padT + plotH - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="12">{v:.1f}%</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{padT+plotH+18}" text-anchor="middle" font-size="11">{name}</text>')
    parts.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(parts))


def dump_qualitative(rows, path):
    qual = [r for r in rows if r.get("tag") != "clean"]
    if not qual:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qual[0].keys()))
        w.writeheader()
        w.writerows(qual)


def main(rj, csvp, prefix):
    rep = json.load(open(rj, encoding="utf-8"))
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    tbl = build_table(rep)
    with open(prefix + "table1.md", "w", encoding="utf-8") as f:
        f.write(f"# Table 1  model={rep['model']}  n={rep['n']}  seed={rep['seed']}\n\n")
        f.write(tbl + "\n")
    print(tbl)
    grouped_bar_svg(rep, prefix + "fig_methods.svg")
    latency_svg(rep, prefix + "fig_latency.svg")
    if rep.get("danger_subset"):
        danger_subset_svg(rep["danger_subset"], prefix + "fig_danger_subset.svg")
        dump_qualitative(rows, prefix + "qualitative_danger.csv")
    print("figures written to", prefix + "*.svg")


if __name__ == "__main__":
    main(*sys.argv[1:4])
