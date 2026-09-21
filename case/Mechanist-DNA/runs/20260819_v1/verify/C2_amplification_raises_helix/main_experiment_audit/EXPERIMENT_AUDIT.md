# Experiment Audit Report — Claim C2

**Date**: 2026-08-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.6-luna)
**Project**: alpha-Helix-Directed DNA Generation via SAE Amplification (Evo2-7B)
**Claim**: C2 — Amplifying the alpha-helix-selective feature set S during Evo2 decoding raises predicted alpha-helix content of translated ORFs vs an identically-decoded unsteered baseline.
**Linked milestones**: M2, M-CTRL, M3

## Overall Verdict: PASS
*This is C2's methodology-integrity verdict (evaluation honesty), not semantic support.*

## Integrity Status: pass

## Checks
### A. Ground Truth Provenance: PASS
ESM-2 probe target = real DSSP labels from AlphaFold-DB structures; readout independent of Evo2/SAE; helix content reported as PREDICTED proxy, explicitly disclosed.
### B. Score Normalization: PASS
Unit decoder directions, defined activation scaling; decoding, seed-pairing, treatment-independent QC, ITT assignment, bootstrap estimands all specified; no self-normalization.
### C. Result File Existence: PASS
results/m3final_analysis.json alpha=1 row matches reported n=2205, d_cond_unpaired=+0.0155 p=0.0325, d_itt(paired)=+0.0146 p=0.0110 (executor-verified).
### D. Dead Code Detection: PASS
Intervention, paired-seed eval, QC/ITT handling, and stats correspond to executed components.
### E. Scope Assessment: PASS
Claim explicitly about PREDICTED helix content; proxy substitution (ESMFold->ESM-2 probe) and its limits disclosed; effect size honestly reported as small (~1.6 pp).
### F. Evaluation Type: synthetic_proxy

## Action Items
- None; methodology honest.
