#!/usr/bin/env python3
"""Phase 6 code review: external LLM review of the C1 model-swap variant."""

import json
import requests

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

diff_content = """## DIFF.md (what changed vs main experiment)

Single change: model = /data/zhenqian/models/Qwen2-Instruct-7B (was Meta-Llama-3-8B-Instruct)
All other hyperparameters unchanged:
- Same AdvBench (520 harmful) + Alpaca (520 matched benign) datasets
- Same seed=0, split 60/20/20 = 312/104/104
- Same evaluation code (m_prep_extract_and_directions.py + m1_claim1_directions.py)
- Same thresholds (AUROC>=0.85, cosine_ratio<=0.5*split_half_reference)

Position offsets: common.py resolve_positions() uses n-6 and n-1 offsets (relative to end).
Verified for Qwen2: tokenized "Hello world test prompt" produces 23 tokens, last 6 are:
[' prompt', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
So n-6 = last user content token, n-1 = the \n after 'assistant'.
Same semantics as Llama-3 (n-6 = last user content token, n-1 = post-instr).

Qwen2 adds system message by default ("You are a helpful assistant."); this adds tokens
to the start, but position offsets from the end (n-6, n-1) remain valid.

Qwen2-7B has 28 hidden layers vs Llama-3-8B's 32. The sweep covers all layers.
"""

prompt = f"""You are a rigorous ML code reviewer checking a model-swap variant for correctness and fairness.

## Claim being tested:
C1 — Two distinct, approximately-linear, independently-recoverable directions in the residual stream of instruction-tuned LLMs (harmfulness direction h and refusal direction r)

## Main-experiment setup:
- Method: difference-in-means + logistic-regression probe AUROC
- Dataset: AdvBench (520 harmful) + Alpaca (520 matched benign), 60/20/20 split, seed=0
- Model: Llama-3-8B-Instruct (fp16, 32 layers)
- Metrics: AUROC(h on harmfulness), AUROC(r on refusal), cos(h,r) vs split-half reference
- Results: partial (h AUROC=0.9998, cos=0.174, refusal-side unmeasurable at 98.7% base refusal)

## Variant diff (what changed):
{diff_content}

## Key evaluation code (common.py resolve_positions):
def resolve_positions(input_ids):
    n = len(input_ids)
    if n < 6:
        raise ValueError(f"prompt too short: {{n}} tokens")
    t_final_instr = n - 6
    t_post_instr = n - 1
    return t_final_instr, t_post_instr

## Review checklist:
1. Does the variant implement ONLY the model swap, nothing else?
2. Are any hyperparameters silently different from the main experiment? List each.
3. Does evaluation still use dataset-provided ground truth labels (not another model's output)?
4. Is the metric computed the same way as the main experiment?
5. Is the position offset logic (n-6, n-1) valid for Qwen2's chat template? (Verified: yes, based on tokenizer inspection above)
6. Any leakage risk introduced by the swap?
7. Does the Qwen2 system message (auto-added by its chat template) change the comparison fairness?

For each issue: CRITICAL / MAJOR / MINOR and the exact fix.
Return JSON with structure: {{"issues": [{{"severity": "CRITICAL|MAJOR|MINOR", "description": "...", "fix": "..."}}], "overall": "APPROVE|REJECT", "notes": "..."}}
Return ONLY valid JSON, no markdown."""

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000},
    proxies={"http": None, "https": None},
    timeout=120
)
print(f"Status: {resp.status_code}")
try:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print("CODE_REVIEW_RESPONSE:")
    print(content)
except Exception as e:
    print(f"Parse error: {e}")
    print(resp.text[:2000])
