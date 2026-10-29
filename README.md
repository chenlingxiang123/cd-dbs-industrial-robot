# CD-DBS: Constraint Decoding + Dual-Branch Self-Check for Safe Industrial Robot Instruction Generation

Official code and data for the study *"面向工业机器人指令生成的约束解码与双分支自检方法"* (Constraint-Decoding and
Dual-Branch Self-Check for Industrial Robot Instruction Generation).

Large language models can translate work orders into industrial robot motion
instructions, but their outputs may violate machine safety rules — out-of-range
speed/force, unreachable poses, forbidden collaboration zones, wrong process
ordering.
This repo implements and benchmarks a **training-free, inference-time safety
framework** on real model outputs:

1. **CD (Constraint Decoding)** — the industrial safety rule table is compiled
   into a character-level grammar trie plus *cross-field joint feasibility*
   checks (range, workspace radius `x²+y²+z² ≤ R²`, collaboration forbidden
   zone, force ceiling, process precedence). Illegal tokens get logit −∞ during
   decoding, so **every fully generated instruction is guaranteed safe**.
2. **DBS (Dual-Branch Self-Check)** — low/high-temperature branches sample the
   same work order; consistency voting over parsed parameters and step
   sequences flags uncertain outputs, with a degradation (refuse-to-send) path.
3. **CD+DBS** — full framework: grammar-safe branches plus consistency vote.

## Headline results (real inference, no simulated generations)

320 procedurally constructed work orders (50 % clean, 50 % *danger* stress
samples: overspeed / overforce / forbidden-zone orders), seed `20261028`,
frozen protocol ([`PROTOCOL.md`](PROTOCOL.md)).

| Method | Danger rate (1.7B) | Violation rate | Safe-exec rate | Danger rate (4B) | Safe-exec rate (4B) |
|---|---|---|---|---|---|
| Baseline (free sampling) | **50.0 %** | 50.0 % | 49.1 % | 46.6 % | 46.9 % |
| + CD | **0 %** | 0 % | 97.5 % | **0 %** | 98.1 % |
| + DBS (free branches) | 50.0 % | 50.0 % | 49.1 % | 46.9 % | 52.2 % |
| + CD + DBS | 0 % | 0 % | 97.5 % | 0 % | 98.1 % |

**Key negative finding**: on Qwen3-1.7B, pure sampling-consistency self-check
detects **0 %** of the true hallucinations — all branches *consistently* copy
the dangerous values, and "consistent" looks "trustworthy" to the voting rule.
On Qwen3-4B the hit rate rises to 81.8 % only because the baseline itself
becomes parse-unstable. Conclusion: syntactic hard constraints and semantic
soft checks are **complementary, neither suffices alone**.

Danger subset (n = 160): baseline unsafe-execution rate 100 % (95 % CI
[100, 100]) on 1.7B and 93.1 % [89.2, 97.0] on 4B; CD corrects 100 % of
conflicted fields into the legal domain.

## Repository layout

```
cd-dbs-industrial-robot/
├── PROTOCOL.md              # frozen experimental protocol (frozen before any
│                            # results; two amendments appended, never rewritten)
├── src/
│   ├── run_cd_dbs.py        # single-file experiment: task generation,
│   │                        # grammar-trie + feasibility masking, DBS,
│   │                        # 4-method comparison, metrics, latency timing
│   ├── smoke_models.py      # model sanity check (schema following, tokenizer)
│   └── make_report.py       # tables + SVG figures from results
├── results/                 # pre-computed results of this paper
│   ├── report_qwen3-1.7b_main_stress.json
│   ├── report_qwen3-4b_robust4b.json
│   ├── samples_qwen3-*.csv  # 640 per-sample rows incl. raw model outputs
│   ├── main_qualitative_danger.csv / robust4b_…  # 160 danger cases each
│   ├── main_table1.md / robust4b_table1.md
│   └── figures/             # SVG figures
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Reproduce

```bash
pip install -r requirements.txt
# weights: any HuggingFace model id (or --model-dir /path/to/local/weights)
cd src
python run_cd_dbs.py --n 320 --model qwen3-1.7b --tag main_stress
python run_cd_dbs.py --n 320 --model qwen3-4b   --tag robust4b
python make_report.py ../results/report_qwen3-1.7b_main_stress.json \
                      ../results/samples_qwen3-1.7b_main_stress.csv \
                      ../results/main_
```

Hardware used in the paper: one consumer GPU (8 GB VRAM, RTX 5060 Laptop),
PyTorch 2.10 + CUDA 12.8, transformers 5.14, local weights. CPU inference
works but is much slower. Seed is fixed at 20261028: same machine, same
library versions → byte-identical work orders and model outputs.

## Data provenance (honesty statement)

- **Work orders**: procedurally generated (no public dataset of
  natural-language work orders → safety-labeled motion instructions exists for
  this domain; the generator in `run_cd_dbs.py` is fully deterministic under
  the fixed seed and is part of this release).
- **Constraint table**: speed full-scale (~1 m/s TCP) and workspace radius conservatively
  aligned to the publicly documented UR5e specifications (reach 850 mm); collaborative
  force ceiling (40 N) derived from ISO/TS 15066 Annex A quasi-static limits (palm 140 N,
  finger 60 N) with a 2/3 safety factor. Keep-out zone placement and process-ordering
  rules remain unit-level engineering assumptions (see PROTOCOL.md section 10).
- **Model generations**: *real* inference outputs (Qwen3-1.7B / Qwen3-4B),
  stored per-sample in `results/samples_*.csv`, including raw texts.
- **Limitations**: safety verdicts come from the constraint solver, not
  physical execution; conclusions are stated within the controllable
  parameterized instruction domain (see `PROTOCOL.md` §6 and §9).

## License

MIT — see [LICENSE](LICENSE).
