# Experiment Audit Report — Claim C1

**Date**: 2026-08-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.6-luna)
**Project**: alpha-Helix-Directed DNA Generation via SAE Amplification (Evo2-7B)
**Claim**: C1 — A small numerically-defined set of Layer-26 SAE latents is alpha-helix-selective on held-out labeled coding DNA.
**Linked milestones**: M1

## Overall Verdict: WARN
*This is C1's methodology-integrity verdict (evaluation honesty), not semantic support.*

## Integrity Status: warn

## Checks
### A. Ground Truth Provenance: PASS
DSSP 3-state labels derive from external AlphaFold-DB structures of real E. coli proteins (RefSeq CDS<->protein<->structure, exact AA agreement); not Evo2/SAE outputs.
### B. Score Normalization: PASS
Tie-corrected midrank AUROC, explicit helix:non-helix ratio and matched beta/coil control bars; no self-normalization by model max.
### C. Result File Existence: PASS
results/m1_features.json matches reported primary_test_auroc=0.6339, CI[0.629,0.639], ctrl_bar=0.543, |S|=20, selection_path=specificity_fallback (executor-verified).
### D. Dead Code Detection: PASS
Reported C1 quantities correspond to the executed selection/analysis path; no material dead code.
### E. Scope Assessment: WARN
Prereg codon-level AUROC>=0.70 NOT met (max 0.633); documented specificity-fallback used and claim explicitly marked 'qualified' -- honest but a material deviation from the primary criterion.
### F. Evaluation Type: real_gt

## Action Items
- Keep the 'qualified' framing; prereg AUROC>=0.70 unmet (documented).
