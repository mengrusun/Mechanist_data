# Experiment Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C2 — Position dissociation: h at t_final-instr, r at t_post-instr
**Linked milestones**: M-prep, M2

## Overall Verdict: WARN
*This is C2's integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Position AUROC uses AdvBench harmfulness labels (dataset-provided). Refusal side uses same proxy as M1 (documented). No GT derived from model outputs without labeling.

### B. Score Normalization: PASS
AUROC computed against dataset labels, no self-normalization.

### C. Result File Existence: PASS
results/m2/claim2_verdict.json exists. Reported crossover_h=-0.00018 matches EXPERIMENT_RESULTS.md. Tracker R003 marked done+neg.

### D. Dead Code Detection: WARN
The crossover-delta logic for refusal-side (crossover_r) is implemented in m2_claim2_position.py but returns NaN because refusal-side split is unmeasurable (same 7-jailbreak limitation). Code exists but cannot fire meaningfully.

### E. Scope Assessment: WARN
The claim posits a "position dissociation" — that h peaks at t_final-instr and r peaks at t_post-instr. However, both positions decode harmfulness at ceiling (~1.000 AUROC), so no dissociation is observable in the data. The 6-position ladder (t_final-instr-2 through t_post-instr+2) was used, which is the planned scope. But the claim language ("position dissociation") implies the result would show clear separation; the actual finding is ceiling-effects at both positions, making the claim's hypothesis untestable rather than confirmed.

### F. Evaluation Type: real_gt
Harmfulness side uses real dataset labels. Refusal side proxy (but labeled).

## Action Items
- Test on a model with lower bare-refusal rate to see if position crossover emerges when refusal-side is measurable.
- Current not-supported verdict is correct and honest — no methodology issue with the negative finding itself.
