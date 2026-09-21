## Claim C3: necessity + sufficiency

### Main experiment (from /auto-experiment)
- Method: activation patching (path patching for necessity, reinsertion for sufficiency), Dataset: k3_chain2_natural (anchor cell, n=500), Model: Mistral-7B-v0.1 → necessity LD=0.955 (PASS), sufficiency LD=0.113 (FAIL)
- Main-experiment verdict: not-supported (C3 requires both necessity AND sufficiency ≥ 0.8; necessity passes, sufficiency fails)

### Variants
| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | Gemma-2-9B | Mistral-7B-v0.1 | Different architecture family, different scale (9B vs 7B), different tokenizer — tests whether the necessity-yes/sufficiency-no pattern is a Mistral-specific artifact or a family-level property of propositional-logic circuits. Result already exists in results/M5_gemma9b.json (M5 milestone, identical pipeline). | EXPERIMENT_RESULTS.md §M5 |

### Reuse declaration
`results/M5_gemma9b.json` was produced by the M5 milestone using the identical activation-patching pipeline (`scripts/cross_family_verify.py`), the identical anchor cell (k3_chain2_natural, n_pairs=500), and the same shortlist selection method (attribution patching screen at 15% cap). The model used is Gemma-2-9B (same model as the model-swap variant). Methodology is equivalent. No fresh run is required. GPU-hours saved: ~0.67 h.

Key variant metrics from M5:
- necessity_recovery.LD = 1.018 (PASS, target ≥ 0.7 relaxed for cross-family)
- sufficiency_recovery.LD = 0.019 (FAIL, far below 0.7 target)
- necessity-yes / sufficiency-no pattern: confirmed on Gemma-2-9B

### Success Criterion (per variant)
Variant is `consistent_with_main_experiment` if it reproduces the same verdict pattern as the main experiment on the frozen C3 claim. Main experiment's verdict on C3 is `not-supported` (because C3 requires BOTH necessity AND sufficiency ≥ 0.8; sufficiency fails). Variant is `consistent` if it also produces verdict `not-supported` — i.e., if sufficiency LD < 0.8 (regardless of necessity). Gemma-2-9B: sufficiency LD = 0.019 < 0.8 → variant verdict = not-supported → consistent = pass.
