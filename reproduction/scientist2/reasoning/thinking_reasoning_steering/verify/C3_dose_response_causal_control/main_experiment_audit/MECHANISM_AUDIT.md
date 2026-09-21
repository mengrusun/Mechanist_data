# Mechanism Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C3 — Adding α·σ·v_b at layer L*(b) causes dose-response behaviour amplification/suppression.
**Linked milestones**: M3 (primary), M1 (v_b provenance)

## Overall Verdict: FAIL
*C3's mechanism-rigor verdict — whether the CAA steering coefficient was swept with the necessary controls to support a dose-response causal claim.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: FAIL
- Triggered: yes — M3 is the core causal-intervention milestone implementing additive steering at L*(b) with coefficient α·σ_proj·u
- Intervention type: CAA (unit-vector contrastive activation addition)
- Sweep grid: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] × σ_proj (7 values; plan's ±3σ dropped after sanity showed collapse)
- σ_proj scaling used: yes (coef = alpha * d["sigma_proj"] in run_M3_steer.py:176)
- Capability metric logged: coherence_rate AND accuracy_rate logged at every α point
- Plateau range: No convincing clean plateau identified. For expressing_uncertainty: α=±0.5σ gives on-target Δrate ≈ +0.017/-0.030 (within ±0.06 binomial noise floor at n=60). α=±2σ shows larger effects but coherence drops to 0.65 (collapse zone). No α window exists where both target effect is robustly above noise AND coherence is preserved.
- Locked α: α_op = 0.5σ (smallest that clears sign check)
- Position in plateau: edge (effect at 0.5σ is within noise at n=60; clean plateau never established)
- Random-direction control: NO — not in code or results
- Sign pattern: preserved for expressing_uncertainty (both signs); broken for generating_validation_examples (positive α reduces rate); undefined for backtracking/self-correction (rates stuck at 0)
- Output-case spot-check: preview_chains in M3 results show coherent chains at α=0.5σ; effect on actual chain text is minimal consistent with the near-zero Δrate. At α=2σ, coherence drops visible in rate (0.65). Metric tracks text at qualitative level.
- Evidence: src/run_M3_steer.py:176 (coef = alpha * sigma_proj), runs/M3_steer/part_A/results_summary.json (analysis.expressing_uncertainty), runs/M3_steer/part_A/cost.json (alphas = [-2,-1,-0.5,0,+0.5,+1,+2])

**FAIL reasons (per catalogue criteria):**
1. The α range [-2,+2] × σ_proj does NOT span 3 orders of magnitude. The range is 4× (from 0.5σ to 2σ is only a factor of 4, and the negative-alpha range mirrors it). The catalogue requires "≥ 3 orders of magnitude (e.g. [0.1, 0.3, 1, 3, 10] × σ_proj)". The experiment's grid spans ±2σ, not the required 3 OOM.
2. No random-direction control at the locked α (α_op = 0.5σ) with n_random ≥ 30. Without this control, it is impossible to distinguish learned-direction specificity from general perturbation effects. This is a direct FAIL criterion.
3. The chosen α (0.5σ) is placed at the plateau edge, not the middle. At 0.5σ, the effect (Δrate = +0.017 for uncertainty) is within the ±0.06 binomial noise floor at n=60. This means the "operating point" may be indistinguishable from noise.
4. For 3 of 4 behaviours, the rates are stuck at 0 across all α — meaning the sweep failed to elicit any dose-response for these behaviours, which is the main C3 claim.

### B–F. Reserved (not_implemented)
Future checks will cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items (to repair C3 mechanism rigor for a future round):
1. Expand benchmark to n ≥ 300–500 tasks to reduce noise floor from ±0.06 to ±0.02–0.03, enabling detection of smaller effects and reliable Spearman ρ estimation.
2. Run a random-direction control at α_op on n ≥ 30 tasks: sample 30+ random unit vectors at L*(b), apply same steering, compare Δrate distribution vs. learned direction.
3. Extend α grid to include smaller values (0.1σ, 0.25σ) and larger values where available before coherence collapse, to better characterize the plateau shape.
4. For backtracking / self-correction / validation-examples: use a benchmark that naturally elicits these behaviours at non-zero baseline rates (current arithmetic-heavy benchmark produces near-0 baseline rates for 3 of 4 behaviours).
