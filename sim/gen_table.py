# -*- coding: utf-8 -*-
"""读取 robot_check_*.csv 明细，聚合生成论文用 Markdown 表 + 拒绝原因分布"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')

TAGS = ['clean', 'speed', 'force', 'zone']
TAG_ZH = {'clean': '合规', 'speed': '超速', 'force': '超力', 'zone': '禁入区'}


def load(model):
    p = os.path.join(RES, f'robot_check_{model}.csv')
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


def agg(rows, tag, which):
    sub = [r for r in rows if r['tag'] == tag]
    n = len(sub)
    ex = sum(1 for r in sub if r[f'{which}_verdict'] == 'EXECUTE')
    rej = n - ex
    reasons = {}
    for r in sub:
        if r[f'{which}_verdict'] == 'REJECT':
            for f in r[f'{which}_faults'].split('|'):
                if f:
                    reasons[f] = reasons.get(f, 0) + 1
    return n, ex, rej, reasons


def main():
    out = []
    out.append('表：机器人端几何/规格执行校验（独立于约束表，采用 UR5e 官方规格与 ISO/TS 15066）\n')
    out.append('| 模型 | 子集 | n | Baseline 可执行 | +CD 可执行 | Baseline 拒绝(原因) |')
    out.append('|---|---|---|---|---|---|')
    for model in ['qwen3-1.7b', 'qwen3-4b']:
        rows = load(model)
        for tag in TAGS:
            n, bex, brej, br = agg(rows, tag, 'base')
            _, cex, crej, cr = agg(rows, tag, 'cd')
            rs = '; '.join(f'{k}×{v}' for k, v in sorted(br.items())) or '—'
            out.append(f'| {model} | {TAG_ZH[tag]} | {n} | {bex/n*100:.1f}% | {cex/n*100:.1f}% | {rs} |')
        # 合计
        n = len(rows)
        bex = sum(1 for r in rows if r['base_verdict'] == 'EXECUTE')
        cex = sum(1 for r in rows if r['cd_verdict'] == 'EXECUTE')
        out.append(f'| {model} | **全部** | {n} | {bex/n*100:.2f}% | {cex/n*100:.2f}% | |')
        out.append('')
    txt = '\n'.join(out)
    with open(os.path.join(RES, 'robot_check_table.md'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
