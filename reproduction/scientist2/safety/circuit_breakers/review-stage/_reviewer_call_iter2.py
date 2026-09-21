"""Iteration 2 reviewer call — the signed-cos M3 rerun engaged the reroute (Δcos_harmful=-0.977,
target ≤-0.30) but over-rotated benign residuals (Δcos_benign=-0.156, target |Δ|≤0.10).
Ask reviewer for the next single-axis fix to bring benign drift back inside gate while keeping
harmful rerouted.
"""
import os, json, sys
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'dmxapi.cn,www.dmxapi.cn'
from openai import OpenAI

client = OpenAI(api_key='<Your_api>',
                base_url='https://www.dmxapi.cn/v1')

# Load reviewer memory from iter1
with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter1.json') as f:
    prev = json.load(f)

CONTEXT = f"""
Iteration 2 of the /auto-iteration-loop for the RR circuit-breakers pipeline. You are the same
reviewer as iteration 1. Recap of iteration 1 (yours):
{json.dumps(prev, indent=2)}

## Iteration-1 outcome (M3+M4 rerun with --rr-loss signed, all other knobs held fixed)

Training (`artifacts/iteration_round_1/m3_signed/train_loss.jsonl` + `training_summary.json`):
- Final L_rr_ema = -0.9995 (cos on train harmful ≈ -1.0 — perfectly anti-aligned).
- Final L_ret_ema = 0.104 (bounded during 0-480; spikes to 2.30 at step 490 as before).
- L_lm_ema = 0.006.
- Grad-norm well-controlled steps 0-460, spikes to 165 at step 490 (identical failure signature).

M4 diagnostic (`artifacts/iteration_round_1/m4_signed/`):
- Δcos(a_h, d_h)     = -0.977  (target ≤ -0.30) ✓ EXCEEDED by 3×
- Δcos(a_h, d_ctrl)  = +2.7e-5 (specificity control) ✓ perfect
- Δcos(a_b, d_h)     = -0.156  (target |Δ| ≤ 0.10) ✗ VIOLATED by 55 %
- criterion_c1_reroute_passed = false (fails on benign side only).

Per-site tuned cos:
  harmful @ sites 9..14: -0.993, -0.995, -0.995, -0.995, -0.994, -0.994
  benign  @ sites 9..14: -0.260, -0.265, -0.273, -0.257, -0.259, -0.250
  benign was -0.10 at base — the reroute is partially bleeding onto benign (0.16 drift).

## Interpretation

The signed-cos loss over-drove: it went from Δcos_harmful = -0.02 (before iter1) all the way to
-0.977 (after iter1) — well past the -0.30 target. The over-shoot itself is not the problem
(harmful being maximally anti-aligned is fine), but the LoRA's degrees of freedom leak onto
benign residuals because L_ret's β=1.0 is too weak relative to α=10 on the aggressive signed
loss. In iteration 1 the harmful reroute achieved cos = -1 while the retention target was
||a_tuned - a_base||² ≤ 0.10 (i.e. β=1.0). We need to bring benign drift from 0.156 to ≤ 0.10
while ideally holding harmful drift somewhere in [-0.90, -0.30] (still overshooting the ≤-0.30
gate but leaving margin).

## What to try next (pick ONE)

A. **Rebalance α vs β**: keep signed loss, drop α to 3.0 or increase β to 3.0 (or both,
   symmetric). Rationale: at α=10 the LoRA burns all its capacity on rerouting, so retention
   loses. This has no code change beyond one CLI flag.

B. **Add a hinge / clip on the signed cos**: `L_rr = max(cos + m, 0)` with margin m ∈ [-0.5, -0.3],
   so once cos falls below -m the loss saturates and stops pushing further, leaving retention
   free to hold benign. Requires a tiny script edit to add a new `--rr-loss` option
   `signed_hinge` and a `--rr-margin` flag.

C. **Restrict LoRA target modules** from all-linears to MLP-only (gate_proj/up_proj/down_proj)
   or Attn-only (q/k/v/o) — reduces the number of degrees of freedom the adapter can spend on
   contaminating benign. Rank stays 16.

D. **Anneal LR / shrink schedule tail** — current cosine schedule to 0 causes step-490 collapse.
   Cap min-LR at 1e-5 and cut training to 400 steps to avoid the tail. Independent of the
   over-reroute issue but eliminates the reproducible instability.

E. **Combined (α↓ + β↑ + LR floor)** — allowed if you argue they must move together.

## Constraint envelope (unchanged)
- Budget: ~7.6 GPU-h remaining of 10 h. Iteration used 0.245 GPU-h so far.
- Claims still pinned. Action must be ② only.

## Your task

Output JSON with same schema as iter1:
- `score` (0..10, current pipeline state after iter1)
- `verdict` (ready/almost/not ready)
- `weaknesses`
- `action_type` (must be "②")
- `fix_axis` (label of the axis you're touching this iter)
- `fix_spec` (concrete CLI flags / code edits)
- `rationale` (≤ 250 words)
- `expected_signal` (what M4 should look like after this fix — both harmful and benign)
- `budget_gpu_hours`
- `still_unresolved_after_fix`
"""

resp = client.chat.completions.create(
    model='gpt-5.4',
    messages=[
        {"role": "system", "content": "You are a rigorous ML research reviewer. Reply with JSON only, no code fences."},
        {"role": "user", "content": CONTEXT},
    ],
    temperature=0.1,
)
raw = resp.choices[0].message.content
print("---RAW---"); print(raw)
s = raw.strip()
if s.startswith("```"):
    s = s.split("```", 2)[1]
    if s.startswith("json"): s = s[4:]
    s = s.rsplit("```", 1)[0].strip()
j = json.loads(s)
with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter2.json', 'w') as f:
    json.dump(j, f, indent=2)
print("---SAVED---")
