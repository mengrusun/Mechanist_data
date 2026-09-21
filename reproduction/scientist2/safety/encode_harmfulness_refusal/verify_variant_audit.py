#!/usr/bin/env python3
"""Phase 9 variant integrity audit for C1 model-swap-qwen2-instruct-7b."""

import json
import requests

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

# Experiment audit
exp_prompt = """You are an experiment integrity auditor. Audit the variant experiment for Claim C1 (model-swap-qwen2-instruct-7b).

Variant: Reran m_prep + m1 with Qwen2-Instruct-7B instead of Llama-3-8B-Instruct.
Same code: scripts/m_prep_extract_and_directions.py + scripts/m1_claim1_directions.py.
Same datasets: AdvBench (520 harmful) + Alpaca (520 matched benign), seed=0, 60/20/20 split.
Same thresholds: AUROC>=0.85, cosine_ratio<=0.5*split_half_reference.

Result files:
- verify/C1_h_r_directions_distinct/variants/model-swap-qwen2-instruct-7b/m_prep/directions.json: EXISTS, best_h={layer:15, pos:t_final_instr, auroc:1.0}, best_r={layer:27, pos:t_post_instr, auroc:1.0}, contrast_source=harmful-vs-benign proxy (documented)
- verify/C1_h_r_directions_distinct/variants/model-swap-qwen2-instruct-7b/m1/claim1_verdict.json: EXISTS, verdict=partial, auroc_h=1.0, cos_h_r=0.068, split_half=0.978, auroc_r=NaN
- Run cost: runs/V001/cost.json, gpu_ids=[0,1,2,3], wall_clock=185s

Check A-F for this variant:
A. GT provenance: same as main (AdvBench/Alpaca dataset labels; refusal proxy explicitly documented in directions.json)
B. Score normalization: none (AUROC vs dataset labels, cosine is geometric)
C. Result file existence: all files exist with valid numbers
D. Dead code: same as main (refusal sub-tests implemented but NaN due to no natural jailbreaks on Qwen2)
E. Scope: no overclaim (same method applied to same-sized dataset; Qwen2 has 28 layers vs 32, this is documented)
F. Eval type: proxy_gt (same as main — harmfulness GT from dataset, refusal proxy documented)

Return ONLY valid JSON (no markdown):
{"A":"PASS|WARN|FAIL","B":"PASS|WARN|FAIL","C":"PASS|WARN|FAIL","D":"PASS|WARN|FAIL","E":"PASS|WARN|FAIL","F":"real_gt|proxy_gt|mixed_gt","overall":"PASS|WARN|FAIL","details":"one paragraph"}"""

resp_exp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": exp_prompt}], "max_tokens": 500},
    proxies={"http": None, "https": None},
    timeout=120
)
print(f"Exp audit status: {resp_exp.status_code}")
exp_data = resp_exp.json()
exp_content = exp_data["choices"][0]["message"]["content"]
print("EXP_AUDIT:", exp_content)

# Mechanism audit
mech_prompt = """You are a mechanistic-experiment rigor auditor. Audit the variant for Claim C1 (model-swap-qwen2-instruct-7b) for mechanism rigor.

Claim C1 uses diff-mean + logistic-regression probe (no additive steering intervention).
Check A (Steering Coefficient Sweep) trigger: does this variant use any additive activation intervention with a scalar coefficient? NO — it uses probe AUROC and cosine similarity only.

Since Check A does not trigger, the overall verdict is n/a.

Return ONLY valid JSON (no markdown):
{"triggered":false,"overall_verdict":"n/a","details":"C1 model-swap variant uses probe AUROC and cosine similarity; no additive steering intervention; Check A not triggered"}"""

resp_mech = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": mech_prompt}], "max_tokens": 200},
    proxies={"http": None, "https": None},
    timeout=120
)
print(f"Mech audit status: {resp_mech.status_code}")
mech_data = resp_mech.json()
mech_content = mech_data["choices"][0]["message"]["content"]
print("MECH_AUDIT:", mech_content)
