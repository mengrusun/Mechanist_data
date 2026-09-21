# Experiment Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C3 — Additive steering dissociates the effects: h flips internal harmfulness readout with refusal unchanged; r flips refusal with harmfulness readout unchanged
**Linked milestones**: M3

## Overall Verdict: WARN
*This is C3's experiment-methodology integrity verdict (separate from mechanism rigor which is in MECHANISM_AUDIT.md).*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Steering targets (harmfulness readout on harmful prompts, refusal rate on benign prompts) use AdvBench labels and string-classifier refusal detection — same transparent proxy as M1. No model-output-derived GT without labeling. Ground truth for the direction itself (h, r) is the same documented proxy.

### B. Score Normalization: PASS
h-readout is the dot-product projection onto the unit-normed h direction — a geometric measure, not normalized by model output. Refusal rate is a binary string-match classifier output — not normalized by model-predicted probabilities.

### C. Result File Existence: PASS
results/m3/claim3_verdict.json exists. 28 steering_metrics.json files under results/m3/{direction}_a{alpha}/ all exist. Numbers in EXPERIMENT_RESULTS.md (e.g., h_delta_max=5.90, r_delta_pos2=0.52, off-target_h_ok=true) match claim3_verdict.json contents. Tracker R004-R019 all marked done.

### D. Dead Code Detection: WARN
The secondary Llama Guard judge path (--secondary_judge flag in m3_claim3_steering.py) is implemented but set to disabled by default. The flag exists in the code but was not activated for main experiment runs. Conservative WARN: the secondary judge metric is not a dead function per se, but was not exercised in any of the 28 cells.

### E. Scope Assessment: WARN
The claim says "h flips internal harmfulness readout with refusal unchanged; r flips refusal with harmfulness readout unchanged." The r-direction effect on refusal is only observable at alpha=+2 (benign refusal jumps from 0.01 to 0.53); at alpha=+1 the effect is absent (0.01). The claim language implies a dose-response dissociation, but r's refusal effect is highly nonlinear (threshold-like at the largest alpha). The h-direction result is clean and monotone. Scope language is appropriate for h; slightly weaker for r. WARN rather than FAIL because the plan acknowledged this pattern and r's alpha=+2 result does demonstrate the effect.

### F. Evaluation Type: mixed_gt
Harmfulness readout: self-supervised projection (geometric); Refusal rate: proxy GT (string classifier on model completions, explicitly documented). Mix of real geometric measure and proxy behavioral label.

## Action Items
- Consider reporting r's dose-response curve more carefully (effect only at alpha=+2) to avoid overclaiming linearity
- Secondary Llama Guard judge could corroborate the refusal rate signal — running it on a 20% subset (as planned) would strengthen C3 evidence
