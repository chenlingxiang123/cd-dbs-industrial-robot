# -*- coding: utf-8 -*-
"""Smoke: does a model follow the 8-line instruction schema and copy gold
values? Also verifies tokenizer pieces needed by the grammar mask
(a standalone "\n" token and single-digit tokens).

Usage:
    python smoke_models.py [Qwen/Qwen3-1.7B Qwen/Qwen3-4B ...]
"""
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cd_dbs import FEWSHOT, gen_task, parse_output, wrong_vs_gold

DEFAULT_MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]

SEED = 20261028
rng = random.Random(SEED)
tasks = [gen_task(rng, "clean") for _ in range(10)]
tasks += [gen_task(rng, t) for t in ("speed", "force", "zone")]

for mid in (sys.argv[1:] or DEFAULT_MODELS):
    name = mid.split("/")[-1]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    V = len(tok)

    def tid(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    print(f"[{name}] token len={V}; '\\n' ids={tid(chr(10))}; "
          f"'X' ids={tid('X')}; '-5' ids={tid('-5')}; digit0={tid('0')}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16).to("cuda").eval()
    ok = wrong = 0
    for body, gold in tasks[:12]:
        prompt = FEWSHOT + body
        inp = tok(prompt, return_tensors="pt").to("cuda")
        torch.manual_seed(7)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=96, do_sample=True,
                                 temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inp["input_ids"].shape[1]:])
        rec = parse_output(text)
        if rec is None:
            print("  PARSE FAIL:", repr(text[:120]))
            wrong += 1
            continue
        w = wrong_vs_gold(rec, gold)
        ok += (not w)
        wrong += w
        print(f"  parsed, wrong={w} act={rec['ACT']} spd={rec['SPD']} "
              f"x={rec['X']} tag={gold['tag']}")
    print(f"[{name}] copy-correct {ok}/{min(12, len(tasks))}, "
          f"load+run {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
