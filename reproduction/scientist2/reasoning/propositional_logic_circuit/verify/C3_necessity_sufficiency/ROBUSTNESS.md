## C3: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  →  PASS

- swap_variants_run: true
- Main-experiment verdict on C3: not-supported
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension: matches the main experiment (consistent=pass, claim_supported=fail while main experiment=not-supported, delta necessity LD +0.063 / sufficiency LD -0.094 — same necessity-yes/sufficiency-no pattern recurs on Gemma-2-9B)

**Reuse note**: the model-swap variant did not require a fresh GPU run. `results/M5_gemma9b.json` (M5 milestone, Gemma-2-9B, identical pipeline + anchor cell) was reused as the model-swap result. GPU-hours saved: ~0.67 h.

Interpretation: The C3 claim (both necessity AND sufficiency ≥ 0.8) is **robustly not-supported** across the model dimension. On Mistral-7B (main experiment): necessity LD=0.955 (PASS), sufficiency LD=0.113 (FAIL). On Gemma-2-9B (model-swap variant): necessity LD=1.018 (PASS), sufficiency LD=0.019 (FAIL). The pattern is strikingly consistent — the shortlist is necessary (patching it onto corrupted prompts nearly fully restores clean behavior in both models) but decidedly not sufficient (reinserting only the shortlist's clean activations onto a resample-ablated clean prompt recovers only 11% on Mistral and 2% on Gemma). This cross-family recurrence of the asymmetry substantially strengthens the negative finding for C3 and is the study's most publishable result. The robustness=1.0 verdict means: the "C3 is not-supported" conclusion is robust across the model dimension.

**Main-experiment integrity caveat (Phase 2 WARN)**: C3's experiment audit returned WARN for a KL recovery metric scaling artifact (baseline KL=0.043, causing recovery_KL=-5.618 in M2; documented in Notes but not elevated to per-claim verdict box). This does not affect the LD- and PD-based verdict on necessity or sufficiency, which are directionally conclusive and unambiguous. The WARN is logged but does not reverse the PASS.
