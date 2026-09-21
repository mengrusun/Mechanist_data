# Experiment Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C2 — Each behaviour direction is extractable from a small pool (n_pairs ≤ 200): split-half cos ≥ 0.7 at every n_pairs ≥ 25, and steering-effect ratio-to-large-pool ≥ 0.8 for every behaviour.
**Linked milestones**: M2 (also M1 for reference direction)

## Overall Verdict: WARN
*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
Steering effect is measured using LLM-judge behaviour annotations (same GPT-5.4 judge). Not model-self-labelled (the judge is an independent strong LLM used as annotator), but it is still proxy evaluation rather than dataset GT. The ratio_to_large_pool is normalized by a reference effect size derived from model outputs under steering — this is a relative metric, acceptable for stability analysis, but not oracle GT.

### B. Score Normalization: WARN
The `ratio_to_large_pool` metric is explicitly normalized by a steering-effect denominator computed from model outputs: `ratio = rate_change(n_pairs) / rate_change(n_pairs=200)`. The denominator is itself a model-output-derived quantity (behaviour rate change under steering). This is a relative stability metric, intended by the claim design, but it technically violates the strict "no normalization by prediction statistics" criterion. The split-half cosine itself is fine (direction cosine, purely geometric).

### C. Result File Existence: PASS
The md claim of split-half cos ≈ 0.588 at n=25 for uncertainty is consistent with the M2 sweep data: the three seeds at n=25 show values 0.521, 0.559, 0.683, whose mean is approximately 0.588. The cos-to-reference of 0.944 at n=25 is consistent with the sweep data (values 0.947, 0.925, 0.947). Other behaviours are listed as untestable (n_pos ≤ 14 for pool < 10), matching the stated corpus caps. Results at runs/M2_smallpool/results_summary.json are non-empty.

### D. Dead Code: PASS
The relevant stability fields in M2 (split_half_cos, cos_to_reference, steered_rate, delta_rate, ratio_to_large_pool) are present and non-empty for expressing_uncertainty in the results file. The sweep iteration over (behaviour, n_pairs, seed) is implemented in run_M2_smallpool.py.

### E. Scope Assessment: PASS
The limitation is clearly disclosed: only expressing_uncertainty is testable (n_pos=28 in extract pool), others are untestable due to n_pos ≤ 14 (below the minimum 10 pairs needed to sweep n_pairs=10). The M2 subset used 30 tasks for steering-effect (plan asked 50; realized 30 for budget; EXPERIMENT_RESULTS.md discloses this). The claim uses "partial / suspected_under_power" verdict — no overclaiming language in the report.

### F. Evaluation Type: synthetic_proxy
Direction stability (split-half cosine) is geometric (no GT needed). Steering-effect ratio is measured via LLM-judge behaviour-rate proxies, not real GT. Classified as synthetic_proxy for the steering-effect component.

## Action Items
- The ratio_to_large_pool denominator being model-derived is noted but acceptable for a relative stability comparison. The paper should clarify this is a relative stability metric, not an absolute one.
- For a future round: expand the auxiliary corpus to get n_pos ≥ 50 for backtracking / self-correction / validation-examples to make the C2 test meaningful for all four behaviours.
