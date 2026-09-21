# Experiment Audit Report — Claim C3

**Date**: 2026-08-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.6-luna)
**Project**: alpha-Helix-Directed DNA Generation via SAE Amplification (Evo2-7B)
**Claim**: C3 — The predicted %-helix dose-response is single-peaked with an identifiable optimum alpha*.
**Linked milestones**: M2, M3

## Overall Verdict: WARN
*This is C3's methodology-integrity verdict (evaluation honesty), not semantic support.*

## Integrity Status: warn

## Checks
### A. Ground Truth Provenance: PASS
Dose-response readout uses the disclosed ESM-2+probe proxy trained/evaluated against external DSSP labels.
### B. Score Normalization: PASS
Alpha grid, fixed decoding, paired seeds, QC/ITT, predicted-%-helix readout consistently specified across the sweep.
### C. Result File Existence: PASS
results/m2_analysis.json matches the 7-point dose-response values and nonflat single-peak analysis incl. alpha_star=1.0 (executor-verified).
### D. Dead Code Detection: PASS
Reported dose-response and held-out reproduction correspond to executed sweep/analysis.
### E. Scope Assessment: WARN
Single-peaked shape established on the 7-point DEV sweep; held-out block reproduces only the rising sub-grid {0,0.5,1,2} and does not independently confirm the descending side/optimum.
### F. Evaluation Type: synthetic_proxy

## Action Items
- Note held-out reproduces only rising sub-grid; descending side is dev-sweep-only.
