# Verify Plan — C4: Cross-LLM+SAE-Pair Generalization

## Claim C4 (frozen)

On Gemma-2-2B + gemmascope-res-16k, SAGE explanations generalize: the predictive- and generative-accuracy gains of SAGE over a reference baseline hold on at least one additional LLM+SAE pair beyond the main pair (Qwen3-4B + transcoder-hp OR GPT-OSS-20B + resid-post-aa).

**Main experiment (M2)**: Qwen3-4B + transcoder-hp (Neuronpedia cached) — SAGE-lite predictive-accuracy only
  - SAGE-lite vs Neuronpedia: Δpearson=+0.153, CI[-0.031,+0.341], p=0.18 → positive trend, not significant
  - SAGE-lite vs GPT5-1shot: Δpearson=-0.055, n.s.
  - Scope: predictive-only + SAGE-lite + 2/3 depths (L28 missing)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | GPT-OSS-20B + resid-post-aa | Qwen3-4B + transcoder-hp | The second verify-pair from task.md's candidate pool, never touched in the main experiment; directly tests cross-pair generalization on a different model family (GPT-NeoX-20B architecture, residual-stream SAE). Available on Neuronpedia (model_id="gpt-oss-20b", sae_id="{layer}-resid-post-aa"). Same protocol as M2 (SAGE-lite + predictive-accuracy via Neuronpedia cached activations). | task.md verify-variant pool; HARD CONSTRAINTS (verify agent) |

## Protocol

Same as M2 (predictive-accuracy via Neuronpedia):
- Fetch Neuronpedia activations for gpt-oss-20b + resid-post-aa features at 3 depths (early L3, mid L11, late L19)
- Seed-lock 80/20 split (split_seed=42)
- Generate 3 explanations per feature: (a) Neuronpedia's public explanation, (b) SAGE-lite (Explainer + Reviewer via GPT-5 DMXAPI), (c) single-pass GPT-5 baseline
- Score each on held-out snippets via GPT-5 scorer → Pearson correlation vs Neuronpedia cached activations
- Target n=15 features/depth = 45 features total (conservative, ~1.5h wall-clock mainly from GPT-5 API calls; GPU used only for Gemma-2-2B forward pass sanity if needed, otherwise purely Neuronpedia+API)
- Since gpt-oss-20b is not locally available and the M2 protocol uses only Neuronpedia cached activations, no GPU-intensive target-LLM forward pass is needed — same as M2 approach

## Success Criterion

For the variant to PASS (consistent with C4's main experiment):
- C4 main experiment verdict = inconclusive-positive-trend (not-supported with positive trend)
- Variant result must be CONSISTENT with this: either also inconclusive/not-supported OR positive-significant
- Specifically: if Δpearson is positive (even if not significant), this is consistent with the main experiment's positive trend direction
- If Δpearson is strongly negative (significant negative), this would be inconsistent (claim FAIL)
- Binary judgment by /result-to-claim: does the variant evidence support the FROZEN claim? (claim_supported = pass/fail)

## GPU budget estimate

- GPT-5 API calls dominate (not GPU): ~45 features × 3 methods × ~20 scored texts + SAGE-lite calls
- GPU used: none for gpt-oss-20b (Neuronpedia cached); CUDA_VISIBLE_DEVICES pinned to subset of {1,2,3,5,6} per task.md constraint but no actual GPU compute needed
- Estimated wall-clock: 1-2h; estimated GPU-hours: ~0.1h (negligible; no target-LLM forward)
