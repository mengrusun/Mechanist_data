"""Iteration 3 — signed_hinge (m=0.3) got Δcos_harmful=-0.572 (great, 2× target) but
Δcos_benign=-0.110 (0.01 over the |Δ|≤0.10 gate). Ask reviewer for the smallest one-axis
tweak to close the 0.01 benign gap without losing harmful reroute."""
import os, json
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'dmxapi.cn,www.dmxapi.cn'
from openai import OpenAI
c = OpenAI(api_key='<Your_api>',
           base_url='https://www.dmxapi.cn/v1')

with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter2.json') as f:
    prev = json.load(f)

CTX = f"""
Iteration 3. Recap of iter-2 (yours):
{json.dumps(prev, indent=2)}

## Iter-2 outcome
- M3 hinge (alpha=10, beta=1, lambda_lm=1, rr_loss=signed_hinge, m=0.30, 400 steps).
- Training clean: L_rr_ema=1.7e-4 (saturated), L_ret_ema=0.036, no tail collapse.
- M4 held-out (128 pairs):
  - Δcos(a_h, d_h)     = -0.572   (target ≤ -0.30) ✓ pass by 1.9×
  - Δcos(a_h, d_ctrl)  = -0.003   ✓ specificity perfect
  - Δcos(a_b, d_h)     = -0.110   (target |Δ| ≤ 0.10) ✗ over by 0.01 (10 % over)
- Per-site tuned harmful cos ≈ [-0.54, -0.61]; per-site tuned benign cos ≈ [-0.20, -0.23]
  (base benign was ≈ -0.10 so drift is 0.10-0.13 per site).

## Interpretation
Very small overshoot — 0.01 above the benign gate. Two candidate one-axis fixes:

A. **Tighten the RR margin m** to 0.20. Currently m=0.30 pushes cos to well past -0.30 because
   batch-averaged saturation still leaks per-sample gradient for samples that happen to have
   cos > -0.30 at that step. m=0.20 keeps the harmful cos target easily achievable but reduces
   how far past it the LoRA rotates.

B. **Increase β from 1.0 to 2.0** (double L_ret weight) — directly punishes benign drift.

C. **Decrease α from 10 to 5** — halves the reroute pressure per sample.

D. **Combined (A + B)**: m=0.25 AND β=1.5 — symmetric small change.

Given how close we are (0.01 miss), a small change on ONE axis should suffice. The cleanest
single-lever is A (tighten m) because it directly targets the observed effect (harmful over-
rotation beyond -0.30) and involves only one hyperparameter.

## Constraint envelope
- Budget: ~7.55 GPU-h remaining of 10 h. Iter used 0.445 GPU-h so far.
- Claims still pinned; action ② only.

## Task
Same JSON schema as before. Pick the single tweak most likely to close the 0.01 benign gap
without regressing harmful reroute. Note that after C1 passes M4, I plan to also re-launch
M5 HarmBench ASR + MT-Bench + MMLU on the new adapter to test whether the reroute translates
into safety improvement (which is what C2 needs). If you think we should just accept the 0.01
miss and move on to C2 evaluation, argue that — that's option E:
E. Accept Δcos_benign=-0.110 as within noise; move on to re-run M5 (HarmBench/MTBench/MMLU)
   and M7 (agent) on the current hinge adapter to test C2/C4 downstream (M6 remains budget-
   gated for VLM).
"""

r = c.chat.completions.create(
    model='gpt-5.4', temperature=0.1,
    messages=[
        {"role":"system","content":"You are a rigorous ML reviewer. JSON only, no code fences."},
        {"role":"user","content":CTX}])
raw=r.choices[0].message.content
print("---RAW---");print(raw)
s=raw.strip()
if s.startswith("```"):
    s=s.split("```",2)[1]
    if s.startswith("json"):s=s[4:]
    s=s.rsplit("```",1)[0].strip()
j=json.loads(s)
with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter3.json','w') as f:
    json.dump(j,f,indent=2)
print("---SAVED---")
