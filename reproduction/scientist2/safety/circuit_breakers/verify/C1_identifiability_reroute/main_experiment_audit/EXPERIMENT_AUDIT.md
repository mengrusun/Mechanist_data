# Experiment Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C1 — Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.
**Linked milestones**: M1, M3 (intermediate — feeds C1b), M4

## Overall Verdict: WARN

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound. Reporting is honest (negative result correctly disclosed), but implementation deviated from plan in several material ways.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS

Ground truth comes from a real labeled dataset (HarmBench + Alpaca paired benign/harmful data). M1 AUC is computed against true binary labels using scikit-learn. M4 compares base vs adapter-applied model on held-out real pairs. No synthetic or self-judged ground truth. Evidence: m1_locate.py lines 49-62 (probe_auc with real labels); m4_diagnostic.py lines 130-141 (base vs tuned comparison on real pairs).

### B. Score Normalization: PASS

AUC calculated with true binary labels — no self-normalization. cos_harmful / delta_cos_harmful are geometric metrics not normalized against model outputs. Specificity control (delta_cos_harmful_ctrl = -0.0003) supports that the measured effect is not a metric artifact. No suspicious proximity to 1.0 that would indicate normalization fraud.

### C. Result File Existence: PASS (with WARN note)

All C1-linked result files exist and are non-empty: artifacts/m1/summary.json, artifacts/m3/training_summary.json, artifacts/m4/activation_drift.json. Numbers in EXPERIMENT_RESULTS.md (AUC 0.9762, delta_cos_harmful=-0.0197) match what's in the artifact files. Tracker status for M1, M3, M4 is "done". WARN: `sweep_notes` in training_summary.json says "L_rr decreasing significantly" but final_L_rr_ema=0.0016 (near-zero). This is misleading static text — the loss was minimized but did not produce the intended representation shift.

### D. Dead Code Detection: PASS

cosine_signed() and cosine_sq() are both defined and both are active (cos_sq_plus_signed default calls both). No unused metric paths in final output. m4_diagnostic.py correctly implements and calls all metric functions that appear in activation_drift.json.

### E. Scope Assessment: WARN

Scope is narrow (single model Llama-3-8B-Instruct, single seed, k=6 sites, 384 train pairs, 128 held-out). This is clearly disclosed — no overclaiming of breadth. However, C1's second half ("can be rerouted") is not established: M4 reroute criterion failed (delta_cos=-0.0197 vs target ≤-0.30, off by ~15×). The "identifiability" part (C1a) has strong evidence (AUC 0.976, 20 mid-late layers above 0.8) but the "rerouting" part (C1b) has a definitive negative. Scope word "identifiable" is supported; "can be rerouted" is not supported by this run.

### F. Evaluation Type: real_gt (supervised representation probing + mechanistic intervention diagnostic)

M1 is a real-data supervised representation-separability test (AUC on labeled pairs). M4 is a held-out mechanistic intervention comparison measuring geometric change before vs after rerouting. Both use real labeled data, not model-generated judges. No simulation-only or self-supervised proxies.

## Implementation Deviations (materially affect interpretation)

1. **Direction type deviation**: Plan (FINAL_PROPOSAL §5.4) specified mean-difference direction `d_h = mean_h - mean_b`. Implementation (m1_locate.py lines 143-149) saves logistic-regression probe coefficient as directions.pt (the primary d_h used by M3/M4). Mean-diff saved separately as directions_meandiff.pt. This is a substantive deviation because it changes what the "harmful subspace direction" is — probe direction has larger cos alignment with individual activations (~0.07 vs ~0.03 for mean-diff), so the choice is pragmatically motivated but not plan-faithful.
2. **Alpha deviation**: Plan FINAL_PROPOSAL §5.1 specified alpha=1.0. training_summary.json shows alpha=10.0.
3. **Loss function deviation**: Plan specified cos_sq. Implementation used cos_sq_plus_signed (cos² + relu(cos)) as default, which adds a hinge toward the negative half-space.
4. **Misleading sweep_notes**: Static string "converged with grad-norm bounded and L_rr decreasing significantly" does not match the actual near-zero EMA (0.0016) throughout the run.

## Action Items

1. Document the probe-direction-vs-mean-diff deviation explicitly in the experiment log; mark as intended deviation with rationale.
2. Fix misleading sweep_notes: derive the string from actual metric values rather than using hardcoded text.
3. Log the actual alpha, loss-function, and direction-type used in the experiment summary rather than the plan defaults.
4. For iteration: run ablations isolating the cause of the reroute failure (direction choice, alpha, loss variant).
