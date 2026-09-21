"""Iteration 1 reviewer call — diagnose root cause + propose ONE main-experiment-script fix.
Output: JSON with {score, verdict, weaknesses, action_type, action_plan, rationale}.
"""
import os, json, sys
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = 'dmxapi.cn,www.dmxapi.cn'
from openai import OpenAI

client = OpenAI(api_key='<Your_api>',
                base_url='https://www.dmxapi.cn/v1')

CONTEXT = r"""
You are an external ML research reviewer for the /auto-iteration-loop. You judge a research
pipeline that trains a Representation-Rerouting (RR) LoRA adapter on Meta-Llama-3-8B-Instruct
to make the model safer while preserving capability, following the "circuit breakers" idea
(without access to the original code — this is a blind reproduction).

The verify stage returned all 4 claims (C1..C4) as INCONCLUSIVE because Phase 2 baseline
integrity gate FAILed. Both experiment and mechanism audits fail. Root-cause evidence:

## M1 (locate sites, PASSED)
- Sites S = [9, 10, 11, 12, 13, 14] chosen (top-6 contiguous mid-band by probe AUC).
- Probe AUC per layer: 9=0.993, 10=0.992, 11=0.993, 12=0.993, 13=0.991, 14=0.991.
- `directions.pt` stores LR-probe coefficient vectors (unit-normalized) — the plan called for
  mean-difference `d_h = mean(a_h) - mean(a_b)`. The script switched to probe coef because
  mean-diff had even smaller cos alignment (~0.03) with individual activations. But even the
  probe-coef direction still has small |cos| on activations (see below).

## M3 (RR LoRA fine-tune — the failure)
- Config: alpha=10.0 (plan said 1.0), beta=1.0, lambda_lm=1.0, rr_loss=cos_sq_plus_signed
  ( = cos^2 + relu(cos) ), LoRA rank=16, alpha=32, target=all linears q/k/v/o + gate/up/down,
  LR=2e-4, cosine schedule, 5% warmup, 500 steps, effective batch=8, 384 pairs, bf16.
- Step 0 L_rr = 0.00121 (cos^2 ~ 1e-3 => |cos| ~ 0.035). Final L_rr_ema=0.00157.
- Grad-norm well-behaved (0.01–1.3) for first 320 steps, then spikes to 852 at step 330,
  159 at step 480, 137 at step 490. L_ret jumps 1e-4 → 2.30 at step 490 (adapter collapse
  in tail of cosine LR schedule).
- The gradient on L_rr is |∂cos²/∂a| = |2·cos · (∂cos/∂a)| which is O(|cos|/|a|).
  Because |cos| starts at 0.035, the effective gradient magnitude is ~35× smaller than
  what a signed-cosine loss would give at cos=0.

## M4 (diagnostic on held-out 128 pairs)
- Δcos(a_h, d_h) = tuned − base = −0.020 (target ≤ −0.30, i.e. 15× too small).
- Δcos(a_h, d_ctrl) [specificity ctrl] = −0.0003 (tuned model didn't move on ANY direction).
- Interpretation: adapter did essentially nothing on residuals. LoRA rank-16 + LR=2e-4 with
  a near-zero-gradient objective produced negligible activation change.

## M6 (Mistral RR — same failure mode)
- Same cos_sq_plus_signed loss, alpha=10.0, 384 pairs, sites also mid-band. Final
  L_rr_ema=0.000222; Δcos_harmful=−0.012.

## Downstream (all consistent with M3 mechanism never activating)
- M5: RR HarmBench aggregate ASR=0.356 (baseline B0 ASR=0.333) — RR is slightly *worse*, as
  expected when the mechanism did nothing.
- M7: harmful_tool_use_rate B0=0.010, RR=0.040 (floor effect + noise from an inactive
  adapter).
- C3 (VLM PGD attack loop) was skipped for budget.

## Constraint envelope
- Claims are pinned by task.md — cannot be softened via action ③. Only action ② (main
  experiment script fix) is on the table.
- Compute budget: ~8 GPU-h remaining of 10 h. A full re-run of M3+M4 costs ~2.3 GPU-h.
- Access: local model at /data/zhenqian/models/Meta-Llama-3-8B-Instruct; conda env lsa_safety.
- Blind reproduction: cannot look up the original RR paper's loss form.

## Candidate fix axes (ONE per iteration; you pick)
1. **Alpha sweep** (α ∈ {1, 10, 30, 100, 300, 1000}) — required by mechanism-audit Check A.
   But naive alpha bumping only helps if the loss surface has real gradient — see
   diagnosis above showing gradient is ~35× smaller than needed.
2. **Loss reformulation** to a *signed* cosine or hinge that has non-vanishing gradient
   at cos ~ 0. Currently the script offers signed=cos itself (drives cos negative) and
   cos_sq_plus_signed (cos² + relu(cos)). But the executed run chose cos_sq_plus_signed,
   whose ReLU(cos) term is ZERO when cos<0 (all steps have cos negative already), so the
   effective loss reduces to just cos². A pure `signed` cos loss (drive cos toward −1) or
   a projection-magnitude loss `||proj_{d_h}(a)||²` (independent of ||a||, gives gradient
   proportional to raw projection, not squared cos) would give meaningful signal.
3. **LoRA rank / target modules** — go to rank 32 or 64; or restrict to MLP-only or
   only-in-sites, in exchange for larger per-layer update capacity.
4. **Site set** — extend from 6 sites to 12 or all-AUC-above-0.9 (would be layers 3–31);
   or switch from post-layer residual to pre-layer to give the LoRA on the layer's own
   projections a direct handle on what site+1 reads.
5. **Data scaling** — 384 pairs → 1000+.
6. Any single combination is allowed if you make the case that they must move together.

## Your task

Output a JSON object (nothing else) with exactly these keys:
- `score` (int, 0..10)  — overall readiness of the current pipeline evidence.
- `verdict` (string in {"ready","almost","not ready"})
- `weaknesses` (list of short strings, ≤ 8)
- `action_type` (must be "②" — main-experiment-script fix)
- `fix_axis` (short label, one of: "alpha_sweep", "loss_reformulation", "lora_capacity",
   "site_set", "data_scale", "combined_<axis1>+<axis2>")
- `fix_spec` (dict with concrete CLI flags / code diffs to run — the reviewer's decision;
   this becomes the M3 + M4 re-run's config)
- `rationale` (string, ≤ 300 words, why THIS fix is expected to unblock the mechanism)
- `expected_signal` (string, what should Δcos_harmful reach in M4 after this fix to
   count as success)
- `budget_gpu_hours` (float, GPU-h to allocate to this fix)
- `still_unresolved_after_fix` (list of claim IDs that would remain uncertain even if
   this fix passes M3/M4; may be empty)

Be sharp. Do not hedge. Pick ONE fix that has the highest chance of moving Δcos_harmful
from −0.02 to ≤ −0.30 in one iteration. Keep the mechanism honest.
"""

resp = client.chat.completions.create(
    model='gpt-5.4',
    messages=[
        {"role": "system", "content": "You are a rigorous ML research reviewer. Reply with JSON only, no code fences, no prose."},
        {"role": "user", "content": CONTEXT},
    ],
    temperature=0.1,
)
raw = resp.choices[0].message.content
print("---RAW---")
print(raw)
print("---PARSED---")
# Strip common code-fence wrappers just in case
s = raw.strip()
if s.startswith("```"):
    s = s.split("```", 2)[1]
    if s.startswith("json"):
        s = s[4:]
    s = s.rsplit("```", 1)[0].strip()
try:
    j = json.loads(s)
    print(json.dumps(j, indent=2))
    with open('/data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers/review-stage/_reviewer_iter1.json', 'w') as f:
        json.dump(j, f, indent=2)
except Exception as e:
    print(f"parse error: {e}")
    sys.exit(2)
