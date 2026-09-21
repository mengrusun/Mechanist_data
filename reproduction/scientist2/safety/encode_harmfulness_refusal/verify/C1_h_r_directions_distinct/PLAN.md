# Verify Plan — Claim C1

## Claim C1: Two distinct, approximately-linear, independently-recoverable directions in the residual stream

**Full statement**: "Two distinct, approximately-linear, independently-recoverable directions in the residual stream: h (harmfulness perception) and r (refusal execution), at different positions."

**Main-experiment verdict**: not-supported (partial — geometry sub-test passes, refusal-side unmeasurable)

### Main experiment (from /auto-experiment)
- Method: diff-mean on contrastive activations (harmful AdvBench vs benign Alpaca), logistic-regression probe AUROC + cosine similarity
- Dataset: AdvBench (520 harmful) + Alpaca (520 matched benign), 60/20/20 split
- Model: Llama-3-8B-Instruct (fp16)
- Result: partial — h AUROC=0.9998, cos(h,r)=0.174 (0.20 of split-half reference); r AUROC=NaN (only 7 natural jailbreaks in 520 bare-harmful attempts → refusal sub-tests unmeasurable)

### Variants (DIMENSIONS=model)

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | Qwen2-Instruct-7B | Llama-3-8B-Instruct | Different model family (Qwen vs Llama) and architecture details; different safety fine-tuning recipe. Qwen2-Instruct-7B is expected to have a lower baseline refusal rate than Llama-3-8B-Instruct on AdvBench prompts, which would make the refusal-side sub-tests measurable and test whether the two-direction geometry (h distinct from r, both independently recoverable) generalizes across instruction-tuned LLM families. | EXPERIMENT_PLAN.md § Verify-Stage Variants; task.md § Resources (verify-stage candidate models) |

### Success Criterion (per variant)

The model-swap variant tests whether the main-experiment's partial verdict (geometry passes, refusal-side unmeasurable) is robust across model families. Claim C1 is partially "not-supported" because refusal-side sub-tests could not run. On Qwen2-Instruct-7B, the question is whether:
- Sub-test (ii) (cosine separation): cos(h,r) is below 0.5 × split-half reference — same direction as main experiment
- Sub-test (i) for harmfulness: h AUROC ≥ 0.85 — same direction as main experiment

Both must hold for the variant to be `pass` (consistent with main experiment's not-supported verdict — specifically, the variant should confirm that geometry is present while refusal-side remains hard). If the variant shows both h and r are independently recoverable with clean AUROCs, that would be BETTER evidence than the main experiment but STILL consistent with the claim (upgrade from partial to supported — which counts as consistent). If the variant shows neither h nor r are extractable, that would also be consistent with not-supported (claim still fails). The variant is `fail` (inconsistent) only if its findings contradict the geometry finding that the main experiment established (e.g., cos(h,r) near 1 on Qwen2 when main experiment showed clear separation, or h AUROC near chance on Qwen2).

**Note**: The main-experiment verdict for C1 is "not-supported" (partial verdict → not-supported in binary scheme). So `consistent_with_main_experiment = pass` if the variant ALSO returns not-supported or partial, and `fail` if the variant returns supported (geometry AND refusal both cleanly measurable and distinct). However, if the variant finds clean refusal AUROCs (because Qwen2 has natural jailbreaks), this is scientific progress — the claim would be closer to supported on Qwen2, which is actually inconsistent with main experiment's not-supported. The robustness verdict reflects this: if Qwen2 supports the claim clearly while Llama-3 didn't, that's fragility in the not-supported direction (the negative finding doesn't hold across models).

**Reduced scale for verify**: run M-prep + M1 only (skip M2 which requires position-crossover testing). Total: ~104 val pairs, one model forward-pass sweep at every layer × 2 positions. GPU budget estimate: ~0.3 GPU-h.
