# Experiment Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via dmxapi, llm-chat-equivalent)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C3 — "The shortlisted components are both necessary and sufficient for the propositional-logic task: necessity — patching their clean activations onto a corrupted prompt recovers ≥ 0.8 on logit_diff / prob_diff metrics; sufficiency — reinstating only their clean activations on an otherwise resample-ablated clean prompt gives ≥ 0.8 recovery; both with specificity gap ≥ 0.6 vs a matched random control set."
**Linked milestones**: M2, M3

## Overall Verdict: WARN

*This is C3's integrity verdict — whether C3's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
GT is loaded from the synthetic dataset JSONL `"answer"` field, set deterministically by `build_propositional_dataset.py` (50/50 True/False balance). In `path_patch_necessity.py` and `reinsertion_sufficiency.py`, GT is obtained via `prop_circuit_lib.compute_metrics` which uses `gt = torch.tensor([1 if r["answer"] == "True" else 0 for r in chunk], device=device)`. Not derived from model outputs.

### B. Score Normalization: PASS
**Necessity (M2)**: Recovery = `(patch_ld - corr_ld) / (clean_ld - corr_ld)`. Denominator is the clean-corrupt logit-diff delta — not the model's own max/mean. Computed per-pair, averaged across n_pairs. No normalization fraud.

**Sufficiency (M3)**: Sufficient-recovery = `(reinsert_ld - ablated_ld) / (clean_ld - ablated_ld)`. Denominator is clean minus fully-resample-ablated behavior — not model max/mean. Control uses same formula on matched random non-shortlist reinsertion. No normalization fraud.

### C. Result File Existence: PASS
All result files cited for C3 exist and match reported values:
- `results/M2_necessity.json`: exists. `recovery.logit_diff=0.9549` (reported 0.955) ✓, `recovery.prob_diff=0.9052` (reported 0.905) ✓, `specificity_gap=0.836` ✓, `success_C3_necessity=true` ✓. n_pairs=500 ✓.
- `results/M3_sufficiency.json`: exists. `sufficient_recovery.logit_diff=0.1129` (reported 0.113) ✓, `specificity_gap=0.1130` (reported 0.113) ✓, `seed_stability_ok=true` ✓, `success_C3_sufficiency=false` ✓. n_pairs=500, n_seeds=5 ✓.
- `results/M5_gemma9b.json`: exists (C3 cross-family replication). `necessity_recovery.LD=1.018` ✓, `sufficiency_recovery.LD=0.019` ✓.
Tracker status for M2 and M3: DONE. ✓

### D. Dead Code Detection: PASS
All key functions invoked for C3 — `run_activation_patch` (for necessity patch and dose-response in path_patch_necessity.py), `run_reinsertion_sufficiency` (for sufficiency reinsertion in reinsertion_sufficiency.py), and `compute_metrics` (for all recovery computations) — produce output fields in `M2_necessity.json` and `M3_sufficiency.json`. The recovery values, dose_response array, control_recovery, specificity_gap, and resample_seed_distribution all appear in results. No dead or decorative metric code found.

### E. Scope Assessment: WARN
**KL recovery metric — scaling artifact disclosed late.** The plan's success criterion for C3 specifies all three metrics (logit_diff, prob_diff, KL) must reach ≥ 0.8 for necessity and ≥ 0.8 for sufficiency. The KL recovery in M2 is `recovery_KL = -5.618` — strongly negative, caused by a known artifact where `KL(clean || corrupt)` baseline is tiny (0.043), so once the patch drives the distribution away from either endpoint the ratio diverges. Similarly in M3, `sufficient_recovery.KL = 0.076` is below the 0.8 target and cannot be fairly compared against it due to the denominator scale issue.

This artifact is **documented in EXPERIMENT_RESULTS.md § Notes** ("KL scaling artifact") and in the M2 milestone entry ("KL recovery goes negative when patch_KL > kl_baseline_clean_corrupt"). The reporting is appropriately restrained: the claim is assessed on logit_diff and prob_diff only, with KL disclosed as unreliable for this baseline scale. However, the **plan's success criterion and C3's claim statement** include KL recovery ≥ 0.8 as a mandatory gate. The disclosure happens in Notes, not in the per-claim verdict box, creating a mild scope discrepancy: the claim as written requires three-metric compliance, but the executed experiment reports a two-metric verdict on necessity (KL excluded by disclosure) and effectively the same on sufficiency. WARN rather than FAIL because the artifact is real, documented, and the two reliable metrics (LD, PD) are directionally conclusive.

C3's n_pairs is 500 (planned: 500) ✓. Specificity control is correctly computed against a matched-size non-shortlist random component set ✓. Resample ablation uses `pool_resample` from the dataset, not zero-ablation ✓. Seed distribution over 5 seeds (planned: ≥ 5) ✓.

### F. Evaluation Type: synthetic_proxy
Task uses a fully programmatically generated propositional-logic dataset. Labels are deterministic, not human-annotated, not from a public benchmark. Classification: `synthetic_proxy`.

## Action Items
1. **KL metric disclosure**: elevate the KL scaling artifact from the Notes section to the per-claim verdict box in EXPERIMENT_RESULTS.md and any write-up. State explicitly: "C3's success criterion requires KL recovery ≥ 0.8; the KL denominator (0.043) is too small for meaningful KL-recovery measurement at this baseline; C3 necessity is assessed on logit_diff (0.955) and prob_diff (0.905) only."
2. Optional: replace KL recovery formula with a more robust divergence measure that doesn't explode under small denominators (e.g., Jensen–Shannon divergence, or KL with a floor on the denominator) to enable the three-metric assessment the plan requires.
