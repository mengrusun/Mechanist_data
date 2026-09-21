# Experiment Audit Report — Claim C3b

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3b — v_c and v_v are causally separable knobs: activation steering along v_c does not preferentially move the v_v probe readout beyond a matched-magnitude random-direction null (and vice versa).
**Linked milestones**: M3 (B3)

## Overall Verdict: WARN

*This is C3b's integrity verdict — whether C3b's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
The primary criterion for C3b is the internal readout (probe projection on the steered residual state). The "ground truth" for whether the probe readout changes is the probe itself — an internal-to-model measurement. The claim explicitly frames this as the PRIMARY metric (internal readout) with secondary corroboration from emitted output (which is real GT for correctness). This is a known limitation of internal-readout-based causal claims and is explicitly pre-registered in the plan. The auditor notes this constitutes dependence on internal model representations rather than external behavioral GT for the primary metric.

### B. Score Normalization: PASS
Steering uses `alpha * sigma_L* * v`, where `sigma_L*` is computed from training activations (elementwise std). This is intervention scaling (the standard practice for steering in units of sigma), not fraudulent score normalization. The criterion (absolute-effect: |Delta| <= 0.5 * sigma_probe_readout) is measured in units of the probe readout distribution's own std — this is appropriate normalization for the test, not circular normalization.

### C. Result File Existence: WARN
`artifacts/steering_results.json` exists and reports all four cross-direction tests PASS via absolute mode. Numbers are consistent with the analysis (sigma_probe_c=1.663, sigma_probe_v=0.362; all Delta_source values << 0.5*sigma threshold). However, methodology concerns: (1) only 3 alpha values tested [-1, 0, +1] — insufficient for plateau identification; (2) single random direction control (not a distribution); (3) the result is a null-steering claim (Delta is SMALL, not large), which is correct for C3b but means the test is somewhat trivially expected from the geometry (|cos|=0.015 guarantees tiny cross-projection).

### D. Dead Code Detection: PASS
`_decide_c3b` function is called and produces outputs in `steering_results.json`. `_run_condition` is called for all direction x alpha x mode combinations. No dead-code concern.

### E. Scope Assessment: PASS
500 test samples, 4500 total forward passes, 3 directions, 3 alpha values, 2 modes (turn1/turn2). Scope is appropriate for the test and matches the pre-registered plan. No scope over-claim language detected.

### F. Evaluation Type: derived
Primary metric: internal probe readout (derived from model internals). Secondary: real_gt for correctness (TriviaQA), c-based for verbalized confidence.

## Action Items
- The single random direction control (n=1) is a weakness — even though the null is expected from geometry, reporting a single control vector is below standard for a causal claim. Consider reporting this as a limitation.
- The 3-point alpha grid is sparse for establishing a behavior-expression plateau — appropriate to acknowledge in paper.
