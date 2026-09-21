"""Iteration 5 — downstream evaluation on RR-iter2 revealed that the mechanism-level reroute
(Δcos_harmful=-0.57) makes the model MORE unsafe, not less. This is a design-level mismatch
between what our reconstructed loss optimizes and what the safety claim needs.
"""
import os, json
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'dmxapi.cn,www.dmxapi.cn'
from openai import OpenAI
c = OpenAI(api_key='<Your_api>',
           base_url='https://www.dmxapi.cn/v1')
with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter4.json') as f:
    prev = json.load(f)

CTX = f"""
Iteration 5. Recap of iter-4 (yours):
{json.dumps(prev, indent=2)}

## Iter-4 outcome — downstream evaluation on RR-iter2 adapter

### C2 (M5): HarmBench + MT-Bench + MMLU
| Metric              | B0 (base) | B1 (adv train) | RR-iter0 (broken) | RR-iter2 (hinge m=0.3) |
|---------------------|-----------|----------------|-------------------|------------------------|
| HarmBench agg ASR   | 0.333     | 0.000          | 0.356             | **0.522** (WORSE)      |
| MT-Bench avg score  | 6.30      | 1.08           | 5.85              | **6.55** (slight ↑)    |
| MMLU accuracy       | 0.58      | 0.54           | 0.57              | **0.53**               |

Per-category HarmBench ASR for RR-iter2:
  direct=0.567, gcg-lite=0.367, persona=0.400, hypothetical=0.600, suffix-injection=0.700,
  human-redteam=0.500.

### C4 (M7): agent function-calling
| Metric                  | B0    | RR-iter0 | RR-iter2 |
|-------------------------|-------|----------|----------|
| harmful_tool_use_rate   | 0.010 | 0.040    | **0.310** (WORSE) |
| BFCL score              | 0.98  | ~0.98    | 1.00     |
Per-category harmful: cybercrime=0.04, disinformation=0.52, fraud=0.52, harassment=0.16.

### The core interpretation problem
Our reconstructed RR loss minimizes cos(a_tuned^h, d_h_base) where d_h is a probe direction
that separates harmful vs benign prompts at sites 9-14 (early-middle layers). We achieved
strong mechanism-level engagement (Δcos=-0.57) with clean specificity (Δcos_ctrl=-0.003)
and preserved MT-Bench capability. But downstream behavior WORSENED — the tuned model is
30-50% more compliant with harmful requests, not less.

**The plausible mechanistic reading**: the probe direction we extracted at layers 9-14 is
the model's own signal for "this prompt is harmful". Rotating harmful activations OFF that
signal effectively erases the model's harmfulness recognition at those early sites, so the
later safety-head layers no longer trigger refusal.

This is a **fundamental design-level issue with our blind reconstruction of the RR objective**,
not a hyperparameter miss. The pattern:
- The safety-claim intent: "reroute to an orthogonal *non-harmful subspace* so the model can't
  generate harmful tokens" — i.e., replace harmful representations with SAFE ones.
- What we actually implemented: "rotate harmful activations away from the direction that USED
  TO predict harmful" — which erases the model's own detection of harm.

## Constraint envelope
- Budget: ~5.8 GPU-h remaining of 10 h. Iteration used ~1.35 GPU-h so far (M3s + M4s + M5 RR-iter2 + M7 RR-iter2).
- Iterations remaining: 2 of 6 (used iter1=signed, iter2=hinge_m03, iter3=hinge_m02, iter4=downstream).
- Claims still pinned. Action ② only (③ full re-entry to claim stage is available but claims
  are pinned so re-entry wouldn't change them — just re-plan an experiment path).

## Iteration 5 options

A. **Move sites later** (e.g. 20-25 instead of 9-14). Rotate residuals AFTER the safety head
   would have fired, so the safety signal survives to trigger refusal, while downstream harmful
   generation is disrupted. Rerun M1 to find/reuse late sites, then M3 with signed_hinge m=0.30.

B. **Reverse the rotation direction**: instead of driving cos → −m, drive cos → +m (align MORE
   with d_h). Push harmful residuals further INTO the harmful subspace so downstream layers
   trigger refusal more reliably. This is counter-intuitive but matches many safety-editing
   papers.

C. **Add a refusal-token forcing term to L_rr**: on harmful inputs, add a CE loss toward a
   canonical refusal continuation ("I cannot help with that."). This is essentially SFT for
   refusal on the harmful side, with the RR reroute term as the mechanism-level regularizer.
   Directly targets the observed pathology (RR moves activations but the model still emits
   harmful text).

D. **Combined C1-adapter + late-layer refusal head**: keep iter-2 adapter, but add a small
   additional LoRA at layers 25-31 trained ONLY on refusal-token forcing on harmful, so the
   overall system has both mechanism-level reroute AND behavior-level refusal.

E. **③ Full claim-stage re-entry** — although claims are pinned in text, re-entering claim
   stage would rebuild the experiment plan and potentially change the M3 objective to
   something like `align tuned harmful activations with a *refusal-direction* proxy`. Would
   consume 1 claim-reentry credit of the 2 available. High cost, high risk in remaining budget.

## Task
Recommend ONE action for iter 5. Given the discovery that the mechanism-level fix doesn't
translate to safety behavior — and we have 2 iterations left — pick what's most likely to
move HarmBench ASR below B0's 0.333 (and ideally close to B1's 0.000) in the remaining
budget.

Same JSON schema. `action_type` can be "②" or "③". If "③", populate `pending_upstream_calls`
in your fix_spec.
"""

r = c.chat.completions.create(model='gpt-5.4', temperature=0.1,
    messages=[{"role":"system","content":"Rigorous ML reviewer. JSON only, no code fences."},
              {"role":"user","content":CTX}])
raw=r.choices[0].message.content
print("---RAW---");print(raw)
s=raw.strip()
if s.startswith("```"):
    s=s.split("```",2)[1]
    if s.startswith("json"):s=s[4:]
    s=s.rsplit("```",1)[0].strip()
j=json.loads(s)
with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter5.json','w') as f:
    json.dump(j,f,indent=2)
print("---SAVED---")
