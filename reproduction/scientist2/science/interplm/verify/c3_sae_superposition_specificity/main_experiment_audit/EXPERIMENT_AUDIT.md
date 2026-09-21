# Experiment Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C3 — The SAE-vs-neuron gap is direct evidence of superposition; SAE coverage strictly exceeds PCA ≈ random-rotation ≈ neurons > shuffled-SAE, with Δ_SAE-PCA ≥ 20 at primary setting.
**Linked milestones**: M3

## Overall Verdict: WARN
*This is C3's integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Same Swiss-Prot external annotation GT as M2. No model-derived GT.

### B. Score Normalization: PASS
- Coverage counts are raw integers under F1 thresholding. No self-normalization.

### C. Result File Existence: PASS
- `runs/m3/superposition_ladder.json` exists and matches EXPERIMENT_RESULTS.md:
  - SAE=15, PCA=0, random_rotation_mean=0.0, neurons=0, shuffled_SAE_mean=0.0
  - Margins all=15 (SAE over each control)
  - order_check_pass=false (strict ordering fails because PCA=neurons=0, tied)
- Tracker row m3: status=done, notes correctly state "ladder direction correct; strict margin Δ_SAE-PCA=15 (near miss of ≥20 threshold)."

### D. Dead Code Detection: PASS
- `runs/m3/superposition_ladder.json` and `runs/m3/summary.md` present. Three seeds for random-rotation and shuffled-SAE arms executed (seeds 42,43,44 folded into one job per tracker). M3 script reuses M2 cached activations.

### E. Scope Assessment: WARN
- The specificity ladder ordering IS supported: SAE > all controls (all controls = 0 at primary τ=0.5).
- The strict margin criterion (Δ_SAE-PCA ≥ 20) is not met at primary setting (Δ=15), but is met at τ=0.3 (Δ=65).
- `order_check_pass=false` because PCA and neurons are tied at 0, which doesn't distinguish the middle of the expected ordering (PCA ≈ random-rotation ≈ neurons). This is a power issue (all controls below detection threshold at primary setting) not an honesty issue.
- 3 seeds for random controls adequately sample the randomness distribution. Honestly disclosed.
- Assessment: WARN (acknowledged scope limitation), not FAIL.

### F. Evaluation Type: real_gt
- Swiss-Prot concept annotations as GT; rule-based F1 measurement.

## Action Items
- At full 10k sequences, controls may score non-zero at primary τ, enabling a cleaner ordering test.
- At τ=0.3 the ladder and margin already robustly support the superposition interpretation.
