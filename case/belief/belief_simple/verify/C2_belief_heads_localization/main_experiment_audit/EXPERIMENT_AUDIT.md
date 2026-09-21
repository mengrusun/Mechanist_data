# Experiment Audit Report — Claim C2

**Date**: 2026-07-22
**Auditor**: self-review (llm-chat MCP unavailable — empty config in .mcp.json)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C2 — Belief Heads Localization: Fisher-information masks yield a smallest H* satisfying all four causal-localization criteria for personal_belief and attributed_belief independently
**Linked milestones**: M2 (M2.1, M2.2, M2.3, M2.4)

## Overall Verdict: PASS

*C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
- Fisher signals computed from log P(y+|x) gradients — ground truth y+ is the dataset-provided gold continuation, not model output.
- Zero-ablation evaluation uses the same dataset gold/distractor pairs as M1.
- Evidence: `scripts/m2_1_fisher.py` (Fisher = (1/N) Σ (∂ log p(y+|x) / ∂θ)² — y+ is dataset GT); `scripts/belief_utils.py:191-235` (zero-ablation eval also uses dataset GT).

### B. Score Normalization: PASS
- Fisher scores are absolute squared-gradient magnitudes summed over parameters — no normalization by model's own max/mean.
- Head candidate scores = fraction of head parameters falling in the mask — relative to the mask count, not model output statistics.
- Jackknife ρ = Spearman rank correlation (dataset-split comparison, not self-normalized).
- Drop measurements: Δ = acc_clean - acc_ablated (raw difference, not normalized).
- Evidence: `scripts/m2_1_fisher.py`, `scripts/m2_2_masks.py`.

### C. Result File Existence: PASS
- Verified from artifact files:
  - `refine-logs/artifacts/hstar/pythia-1b/H_attributed.json`: status=localized, hstar_heads=[[4,1]], hstar_size=1 ✓
  - `refine-logs/artifacts/hstar/pythia-1b/H_attributed_acceptance.json`: drops.target=0.4625 ≈ 0.463 ✓, drops.ppl_ratio=1.011 ✓, all 4 criteria True ✓, mean_rh+2σ=0.257 ✓ (matches EXPERIMENT_RESULTS.md)
  - M2.4 controls: n_controls=20 for both random_head and random_mask ✓
  - All M2 tracker rows show status `done`.
- EXPERIMENT_RESULTS.md claimed values cross-checked against acceptance JSON — values consistent.
- Evidence: `refine-logs/artifacts/hstar/*/H_{target}_acceptance.json` per localized (model, target) pair.

### D. Dead Code Detection: PASS
- M2.1: Fisher accumulation loop, per-head aggregation, jackknife half-tensors — all computed and saved.
- M2.2: Mask construction, ranked-head output — used by M2.3.
- M2.3: Greedy-add + greedy-remove search — produces H* used by M2.4 and M3/M4.
- M2.4: Random-head and random-mask control loops — results used in C2b acceptance test.
- `scripts/m2_4_acceptance.py` aggregation script produces acceptance JSON — confirmed present.
- No unused evaluation paths detected.

### E. Scope Assessment: PASS
- 5 of 6 possible (model, target) pairs tested (pythia-410m × attributed excluded by M1 above-chance gate, per protocol).
- Claim language: "5 of 5 admissible pairs LOCALIZED", "separately for personal_belief and attributed_belief" — precisely scoped. No overclaim.
- |H*| ∈ {1, 2, 4} — smallest sets reported honestly; negative off-belief drops noted ("Ablating H*_personal actually IMPROVED attributed-belief accuracy").
- 20+20 controls per H* (verified from acceptance JSON: n_controls=20 for each).
- Evidence: EXPERIMENT_RESULTS.md M2 section; EXPERIMENT_TRACKER.md M2.3-M2.4.

### F. Evaluation Type: real_gt
- All zero-ablation evaluations use dataset-provided gold/distractor continuations.
- Fisher signals computed from dataset GT gradients.
- Classification: **real_gt** throughout.

## Action Items
None — no integrity concerns identified.
