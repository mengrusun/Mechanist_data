# Experiment Audit Report — Claim C3 Variants

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via dmxapi, llm-chat-equivalent)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C3 — necessity + sufficiency
**Variant audited**: model-swap-gemma2-9b (reuses results/M5_gemma9b.json)
**Linked milestones (variant)**: M5

## Overall Verdict: PASS

*This is the variant-level integrity verdict for C3's model-swap-gemma2-9b variant.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
GT in the variant is loaded from the same synthetic dataset JSONL `"answer"` field as the main experiment, from the same anchor cell (k3_chain2_natural) built by `build_propositional_dataset.py`. GT = deterministic labels, not derived from model output. In `cross_family_verify.py` (M5), the evaluation follows the same `prop_circuit_lib.compute_metrics` path used in M2/M3.

### B. Score Normalization: PASS
Necessity recovery: same formula `(patch_ld - corr_ld) / (clean_ld - corr_ld)` as M2. Sufficiency recovery: same formula `(reinsert_ld - ablated_ld) / (clean_ld - ablated_ld)` as M3. Denominators are clean-corrupt and clean-fully-ablated logit-diff deltas respectively — not model max/mean. No normalization fraud.

### C. Result File Existence: PASS
`results/M5_gemma9b.json` exists on disk. All needle values for C3 judgment are present:
- `necessity_recovery.logit_diff = 1.0178` ✓
- `necessity_recovery.prob_diff = 1.0116` ✓
- `sufficiency_recovery.logit_diff = 0.0190` ✓
- `sufficiency_recovery.prob_diff = 0.0049` ✓
- `n_pairs = 500` ✓
- `cell = "k3_chain2_natural"` ✓
- `anchor_accuracy = 0.96` ✓
- `timing.total_s = 2397.2` (consistent with 0.666 GPU-h) ✓
Model file: `model = "/data/zhenqian/models/LLM-Research/gemma-2-9b"` (downloaded via ModelScope, confirmed in EXPERIMENT_RESULTS.md §M5). ✓

### D. Dead Code Detection: PASS
`cross_family_verify.py` calls all sub-routines relevant to C3: attribution screen (produces shortlist_heads/mlps, sparsity_fraction, cumulative_effect), necessity patch (produces necessity_recovery.*), sufficiency reinsertion (produces sufficiency_recovery.*). All produce populated output fields in M5_gemma9b.json. Role-dissociation output (role_S_matrix, median_dominance_ratio, role_dissociation, block_partition) is also present (bonus, for C1/C2 cross-family check; not needed for C3 judgment). No dead metric code.

### E. Scope Assessment: PASS
All C3-relevant scope parameters:
- n_pairs = 500 (same as main experiment) ✓
- Anchor cell = k3_chain2_natural (same) ✓
- Shortlist fraction = 15% cap (same method; 107 components for Gemma-2-9B vs 158 for Mistral-7B — expected difference due to different model sizes, both hit the 15% cap) ✓
- Batch size = 2 (adapted for Gemma-2-9B's larger memory footprint; doesn't affect metric correctness, only throughput) ✓
- No scope overclaim in EXPERIMENT_RESULTS.md §M5 — results are restrained ("cross-family FAIL on recurrence, but necessity-yes/sufficiency-no pattern recurs") ✓

The role-dissociation sub-component uses top-20 of 107 components (abbreviated) but this is not part of C3's judgment — C3 tests necessity and sufficiency on the full shortlist, which is correctly done at 107 components.

### F. Evaluation Type: synthetic_proxy
Same dataset as the main experiment. Classification: `synthetic_proxy`.
