# Mechanism Audit — C2 (model-swap-olmo-1b)

**Date:** 2026-07-22  
**Auditor:** self-review  
**Claim:** C2 — Belief heads are localizable (H* satisfying all four criteria exists in the model)  
**Variant:** model-swap-olmo-1b (OLMo-1B-hf)

## Overall Verdict: WARN

## Checks

### Check A — Steering Coefficient Sweep: N/A (treated as PASS)

The mechanism intervention is **zero-ablation**: head output `x[..., h*hd:(h+1)*hd]` is scaled by 0.0 (zeroed). No tunable steering coefficient exists; there is no alpha parameter to sweep. N/A per SKILL.md convention.

### Check B — Cross-Vocabulary PPL: WARN

PPL is computed on Pythia-tokenized eval tokens (max ID 50276) applied to OLMo-1B's vocabulary (50304 entries). Since max Pythia token ID (50276) < OLMo vocab size (50304), no index-out-of-bounds error occurs and the forward pass is numerically valid.

**Issue:** Absolute PPL values (10.782 for clean OLMo-1B) are NOT directly comparable to Pythia-1B absolute PPL, as different tokenizations produce different per-token loss scales.

**Mitigation:** The C2d criterion uses the RATIO `ablated_ppl / clean_ppl` — both computed in OLMo's tokenizer-space on the same token sequence. The tokenization mismatch affects numerator and denominator identically, so the ratio remains a valid measure of ablation-induced PPL degradation. The 1.05× threshold on the ratio is model-agnostic and correctly applied.

**Verdict:** WARN — ratio-based C2d is internally consistent; absolute PPL should not be cross-model compared.

### Check C — Hook Correctness: PASS

OLMo-1B (OlmoForCausalLM) uses separate `q_proj`, `k_proj`, `v_proj`, `o_proj` projection matrices per layer. The scaling hook is registered as `forward_pre_hook` with `with_kwargs=True` on each head's associated `o_proj`, targeting `x[..., h*hd:(h+1)*hd] *= s` where `hd=128` (head_dim from `hidden_size=2048 / n_heads=16`).

This correctly scales each attention head's contribution to the output projection before the linear transformation — the exact analogue of Pythia's dense-head hook that targets rows of the combined projection matrix. For `s=0` (zero-ablation), this completely zeroes head h's causal contribution to the residual stream.

### Check D — Fisher Aggregation: PASS

For OLMo's split projections, Fisher score for head h aggregates:

```
score(l, h) = Σ F_q_proj.weight[h*d:(h+1)*d, :] 
            + Σ F_k_proj.weight[h*d:(h+1)*d, :]
            + Σ F_v_proj.weight[h*d:(h+1)*d, :]
            + Σ F_o_proj.weight[:, h*d:(h+1)*d]
```

where d=128 (head_dim). This captures all four projection matrices that exclusively encode head h's computation. No double-counting; no missing projections. Consistent with the main experiment's Fisher aggregation intent applied to OLMo's architecture.

### Check E — M1 Gate: PASS

Above-chance gate correctly applied prior to all Fisher computation:

| Target | Acc | CI_low | CI_high | Admitted |
|---|---|---|---|---|
| personal_belief | 0.778 | 0.746 | 0.808 | yes |
| attributed_belief | 0.731 | 0.697 | 0.763 | yes |
| world_knowledge | 0.930 | 0.889 | 0.956 | yes (Fisher signal only) |

Both belief targets pass M1 gate. Fisher/greedy search only executes for admitted targets.

### Check F — Controls: PASS

20 random-head controls implemented with seeds 100-119. Each control samples a random set of heads (same cardinality as the candidate H*) and evaluates target accuracy drop using the same `eval_hstar` infrastructure (with cached clean baselines for efficiency). C2b criterion: `ablated_target_drop > mean + 2σ(random_head_drops)`. Correctly implemented.

Note: The main experiment also ran supplementary random-mask controls (ablating random parameter subsets), but these are not replicated in the variant. C2b only specifies random-head controls; the random-mask analysis was supplementary. Not a protocol violation.

### Check G — Memory Optimization: PASS

Fisher memory management verified from log output:
- After mask construction: `del F_tgt, F_a, F_b` → GPU 14.8GB → 7.1GB
- After last target's Fisher: `del F_know` → GPU 7.1GB → 2.4GB
- Greedy search for attributed_belief runs at 2.4–6.8GB (well within GPU budget)

## Summary

The OLMo-1B variant correctly implements the four-criteria localization protocol. Hook adaptation to OLMo's split projection architecture is correct. Fisher aggregation covers all head-specific parameters. The cross-vocabulary PPL WARN is a known architectural difference: the C2d ratio criterion remains valid despite different tokenizations; absolute PPL values should not be compared cross-model.

**integrity_status: warn** (eligible for robustness computation — WARN does not disqualify a variant)
