# Experiment Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via dmxapi, llm-chat-equivalent)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C1 — "A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model (target |shortlist|/|total| ≤ ~15% with completeness ≥ 0.9 and single-removal minimality drop ≥ 0.05 on Mistral-7B)."
**Linked milestones**: M1

## Overall Verdict: WARN

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Ground truth is loaded directly from the dataset JSONL `"answer"` field, set deterministically by `build_propositional_dataset.py` (50/50 True/False balance enforced programmatically). Not derived from model outputs.

Code: `gt = torch.tensor([1 if r["answer"] == "True" else 0 for r in clean_chunk], device=device)` (prop_circuit_lib.py)

### B. Score Normalization: PASS
Completeness recovery formula: `(patch_ld - corr_ld) / (clean_ld - corr_ld)`. Denominator is the clean-corrupt logit-difference delta, NOT the model's own max/mean output. No score-normalization fraud pattern found.

### C. Result File Existence: PASS
`results/M1_attribution.json` exists on disk. All claimed values match file contents:
- shortlist_size: 158 ✓
- sparsity_fraction: 0.14962 (reported 0.150 — consistent) ✓
- completeness: 0.9549 (reported 0.955 — consistent) ✓
- completeness_prob_diff: 0.9052 (reported 0.905 — consistent) ✓
- minimality.avg_single_removal_drop: 0.00141 (reported 0.0014 — consistent) ✓
- cumulative_effect: 0.758 ✓
- n_pairs: 500 ✓
- success_C1: False ✓ (correctly reflects partial result)
Tracker status: DONE.

### D. Dead Code Detection: PASS
All metric functions (attribution_scores, run_activation_patch, build_shortlist) are called in the attribution_screen.py pipeline and produce outputs present in M1_attribution.json. No dead or decorative metric code found.

### E. Scope Assessment: WARN
**Undisclosed minimality sampling**: the minimality sweep samples only 20 of 158 shortlist components (`n_min = min(20, len(components))` in attribution_screen.py lines ~165-166) to "save budget". This approximation is NOT disclosed in EXPERIMENT_RESULTS.md's C1 section or in results/final_report.md. The claim references `single-removal minimality drop ≥ 0.05` as a criterion, but the reported estimate of 0.0014 is based on partial sampling (20/158 = 12.7% of the shortlist). The reporting language itself is appropriately restrained ("partial", avoids "comprehensive"/"robust"), but the undisclosed approximation weakens evidential integrity for the minimality sub-claim.

Shortlist cap truncation (cumulative 0.758 vs 0.9 target) IS disclosed and correctly reported.

### F. Evaluation Type: synthetic_proxy
Task uses a fully programmatically generated propositional-logic dataset with deterministic labels. Not a real-world benchmark. No official benchmark eval script exists (custom task). Classification: `synthetic_proxy` (programmatic GT, not human-annotated, not a public benchmark).

## Action Items
1. **Disclose minimality sampling**: add to EXPERIMENT_RESULTS.md and final_report.md that minimality.avg is estimated from 20 of 158 components (step-sampled). The full sweep would require ~138 more component-patch evaluations.
2. Optional: re-run minimality sweep on all 158 components to confirm or refine the 0.0014 estimate — unlikely to change the FAIL conclusion given the consistent low drop across all 20 sampled components.
