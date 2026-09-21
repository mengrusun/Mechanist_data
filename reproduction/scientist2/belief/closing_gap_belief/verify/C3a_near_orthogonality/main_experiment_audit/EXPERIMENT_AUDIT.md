# Experiment Audit Report — Claim C3a

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3a — The gold-correctness direction v_c and the verbalized-confidence direction v_v at the primary reporting layer L* are geometrically near-orthogonal (|cos| <= 0.3 with 95% CI upper bound < 0.4), robust across L*+-2 neighborhood.
**Linked milestones**: M2 (B2), M6 (B7)

## Overall Verdict: PASS

*This is C3a's integrity verdict — whether C3a's experimental process is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
C3a is a derived analytical quantity: cosine between the L2-normalized weight vectors of two trained linear probes. There is no "ground truth" in the traditional sense — the claim is about the geometric relationship between two learned directions. The evaluation is explicitly described as derived from probe weights, not from a separate GT source.

### B. Score Normalization: PASS
No suspicious normalization against model prediction statistics. Cosine similarity is a standard geometric quantity normalized by vector norms (not model output statistics). The random-direction null provides a valid theoretical comparison (expected |cos| ~ 1/sqrt(D) ~ 0.016 for D=4096).

### C. Result File Existence: PASS
Referenced files exist and numbers are internally coherent:
- `cos_bootstrap.json`: mean=0.015, CI=[0.001, 0.034], n=200 -- well below the 0.3 threshold, CI upper 0.034 < 0.4 threshold
- `cos_trajectory.json`: abs_cos_at_Lstar=0.015, neighborhood_L*+-2_mean=0.025 -- both below 0.3
- `nulls.json`: random null mean=0.011, shuffled-label cos=0.016 -- both consistent with theoretical expectations
- `single_pass.json`: |cos|=0.139 (single-pass variant, still < 0.3)
All numbers reported in EXPERIMENT_RESULTS.md match the artifact files.

### D. Dead Code Detection: PASS
Cosine computation (`_cos` function in step4_probes.py) is called in the layer sweep and the bootstrap; outputs appear in all expected artifact files. No dead-code concern.

### E. Scope Assessment: PASS
C3a scope is clear: 32 layers + embedding, 200 bootstrap resamples at L*, neighborhood L*+-2. No scope over-claim language detected. The single-pass variant (B7) is reported alongside as a robustness check.

### F. Evaluation Type: derived
Pure analytical derivation from probe weight vectors. No external GT required.

## Action Items
None. C3a passes all checks cleanly.
