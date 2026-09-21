# Verify Plan — CM: residual direction carries emotion identity and causally modulates accuracy

## Claim CM: statement

Some low-rank residual-stream direction on Qwen3-14B carries the emotional-frame identity (Location: probe accuracy ≥ 0.5 on 6-way emotion identity at some layer vs ≤ 0.2 length-controlled baseline) and causally modulates per-item GSM8K Δaccuracy (Intervention: patching Δaccuracy sign matches C1 on ≥ 60% items, steering dose-response monotone on ≥ 3/4 sites, filler-control Δ near zero, off-target near zero).

## Main experiment
- Method: Probing / Residual-Stream-States (M5) + Causal Attribution / Patching + Representation / Steering-Vectors (M6), Qwen3-14B transformers + vLLM
- Dataset: GSM8K (CoT, 500 items for M2; 50 items for M6)
- Model: Qwen3-14B (bf16)
- Main-experiment metric: probe acc=1.00 (vs null=0.28) at layers 4-36; steering Δacc flat within 4pp noise floor (not-supported at tested scale)
- Main-experiment verdict: not-supported (binary scheme; Location arm supported but Causal arm not supported; combined claim fails)

## Dimensions scope
Active: model (DIMENSIONS=model per /auto-verify invocation)
Excluded: method, dataset (not tested in this verify pass)

## Variants

| # | Dimension | Swap (replaces) | Justification | Expected if claim holds | Expected if claim fails | Risk / confound control | Trust rank | Source |
|---|-----------|-----------------|---------------|-------------------------|-------------------------|-------------------------|------------|--------|
| 1 | model | Qwen3-8B (← Qwen3-14B) | Same Qwen3 architecture family, 8B vs 14B — tests whether the emotion-identity linear direction is a general property of the Qwen3 model family or specific to the 14B scale. Qwen3-8B is on disk at /data/zhenqian/models/Qwen3-8B. Budget-feasible at 50 items × 6 conditions. | Probe accuracy on 6-way emotion identity should be similarly high (>0.7) at some early layer; length-controlled null should remain low (~0.2). If the Location claim holds robustly, the emotion direction should be present at all scales. | Probe accuracy near chance (≤0.3) at all layers on Qwen3-8B, suggesting the emotion identity encoding is scale-specific to 14B or specific to the training checkpoint. | Same prefix corpus (data/prefixes/prefixes.json), same GSM8K items, same layer sweep (layers 0,4,8,12,16,20,24,28,32,36), same logistic probe procedure; only the model checkpoint changes. No need to re-run causal arm to test Location robustness. | 1 | IDEA_REPORT.md (Qwen3 family), NOTICE (task.md verify candidates note same-family siblings) |

## Success Criterion
The variant's claim_supported verdict is judged by /result-to-claim against the frozen CM claim statement. For the model-swap on Location only: if Qwen3-8B probe accuracy significantly exceeds the length-controlled null at some layer (e.g., >0.7 vs <0.3), the Location sub-claim holds on this model, contributing pass to robustness. If probe accuracy is near chance across all layers, the Location sub-claim is model-specific and the variant returns fail (consistent with the Causal arm being not-supported = the combined CM claim is fragile).

## Candidate Pool (audit trail)

### Model candidates (harvested from project context and disk)
| # | Name | Source | Notes |
|---|------|--------|-------|
| Mdl1 | Qwen3-8B | /data/zhenqian/models/Qwen3-8B (on disk) | Same Qwen3 family, 8B vs 14B; same tokenizer family; budget-feasible |
| Mdl2 | Qwen3-4B | /data/zhenqian/models/Qwen3-4B (on disk) | Even smaller; more aggressive scale reduction; architecture comparable |
| Mdl3 | Qwen2.5-14B-Instruct | /data/zhenqian/models/Qwen2.5-14B-Instruct (on disk) | Different Qwen generation (2.5 vs 3); similar scale; tests generation-specificity |
| Mdl4 | Llama-3.1-8B | /data/zhenqian/models/Llama-3.1-8B (on disk) | Different model family entirely; stronger test but higher confound risk (different tokenizer, different training) |

**Selected: Qwen3-8B** — same family, on disk, budget-feasible, cleanest confound control. Reviewer rationale: within-family scale swap is the strongest test of scale-dependence without introducing architecture/tokenizer confounds. Qwen3-4B also on disk but at only 4B parameters the architecture depth changes significantly. Qwen2.5-14B-Instruct tests generation-specificity which is a different question. Llama-3.1-8B introduces too many confounds for a single-variant verify pass.
