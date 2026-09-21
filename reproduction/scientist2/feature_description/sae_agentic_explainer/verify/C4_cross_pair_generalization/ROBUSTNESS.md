# Robustness Report — C4: Cross-Pair Generalization

**Claim ID**: C4
**Claim**: SAGE explanations consistently outperform reference baselines (Neuronpedia, single-pass GPT-5) across architecturally distinct model+SAE pairs on predictive accuracy (Pearson r on held-out Neuronpedia activations).

**Stage 1 (Phase 2) — Baseline Integrity**: WARN
- EXPERIMENT_AUDIT: WARN (Check C: L28 missing from M2 data; Check E: scope narrowed to SAGE-lite + predictive accuracy only)
- MECHANISM_AUDIT: N/A (no additive intervention)
- Combined: WARN → admitted to Stage 2

**Stage 2 (Phase 3–7) — Variant**: model-swap-gpt-oss-20b
- Swap: model Qwen3-4B + transcoder-hp → GPT-OSS-20B + resid-post-aa; layers 8/16/28 → 3/11/19
- N features: 45 (15/layer)
- Protocol: SAGE-lite (Explainer + Reviewer) → predictive accuracy on held-out Neuronpedia activations

**Stage 3 (Phase 8–10) — Verdict**

| Variant | sage_lite pearson | neuronpedia pearson | delta | CI 95% | significant | verdict |
|---------|-------------------|---------------------|-------|---------|-------------|---------|
| model-swap-gpt-oss-20b | 0.1365 | 0.1817 | −0.0451 | [−0.210, +0.115] | no | consistent (not-supported) |

**Robustness computation:**
- N_run = 1, N_eligible = 1 (variant integrity: PASS)
- n_pass = 1 (variant agrees with main experiment verdict: both say "not-supported")
- robustness = 1/1 = 1.00
- ROBUSTNESS_THRESHOLD = 0.50
- 1.00 ≥ 0.50 → **PASS**

**Terminal state**: PASS
**Final verdict**: C4 is robust — the "not-supported" conclusion holds on a third, architecturally distinct model+SAE pair.

## Per-layer breakdown (GPT-OSS-20B variant)
| Layer | sage_lite pearson | neuronpedia pearson | gpt5_1shot pearson |
|-------|------------------|---------------------|-------------------|
| L3 | 0.2307 | 0.3338 | 0.1220 |
| L11 | 0.0140 | 0.0112 | 0.0945 |
| L19 | 0.1649 | 0.2000 | 0.1169 |

## Interpretation
The GPT-OSS-20B variant shows that SAGE-lite (delta_pearson = −0.045 vs neuronpedia) does NOT outperform the Neuronpedia reference baseline on this third pair. This is consistent with the main experiment (M2: Qwen3-4B, delta_pearson = +0.153, CI=[−0.039,+0.391]) which also failed to reach significance. The negative result is robust: across both tested pairs, SAGE-lite does not show a statistically significant improvement in predictive accuracy over Neuronpedia's reference explanations. Note: SAGE-lite does slightly beat gpt5_1shot (delta=+0.025) but also without significance.

## GPU pin verification
- cost.json gpu_ids: [1]
- Allowlist {1,2,3,5,6}: GPU 1 ∈ allowlist — VERIFIED
