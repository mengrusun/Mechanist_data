# Mechanism Audit Report — Claim C3 (RE-AUDIT after iteration-② fix)

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C3 — The amplification-induced α-helix rise is specific to the α-helix feature set S (vs matched-control / β-sheet off-target); S is a causally manipulable, specific knob.
**Linked milestone**: M3 (with M2 providing the fuller dose-response grid)
**Supersedes**: `MECHANISM_AUDIT.pre_iteration_fix.md` (prior verdict **FAIL**).

## Overall Verdict: WARN
*Was FAIL. The iteration-loop type-② mechanism-harness repair removed both FAIL triggers: (1) the
decisive dose is re-locked from α=16 (capability-crashed, valid-ORF 0.888→0.667) to the
capability-preserved mid-plateau dose α=8 (valid-ORF 0.888 ≈ baseline); (2) a genuine random-direction
control (33 independent, norm-matched directions) was added at the locked dose. Residual WARN items are
methodological refinements, not rigor failures.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — additive SAE-feature steering, `code/mechanism.py:36-44` (`delta = self.alpha * self.base_dir`).
- Intervention type: SAE feature steering (additive decoder-direction amplification at blocks.26.post_norm).
- Sweep grid: M2 α∈{-2,0,1,2,4,8,16,32}×3 seeds (includes negative-α sign check + α=0 baseline); M3 α∈{0,4,8,16}×3 arms×3 seeds; random-direction null at α=8 × 33 directions.
- σ_proj scaling used: **no** — α is a raw multiplier on `base_dir` (s_f = mean nonzero SAE activation), not expressed as k·σ_proj. *(WARN item.)*
- Capability metric logged: **yes** — `valid_orf_rate` + pLDDT at every dose. α=0: valid-ORF 0.872; α=8: 0.894 (preserved); α=16: 0.652 (crash).
- Plateau range: capability-preserved region **α ≤ 8**; effect first clearly above baseline noise at α=8 (α=4 is +0.011, within noise; α=8 is +0.096). Collapse at α=16.
- Locked α: **8** — position in plateau: **edge** (upper end of the capability-preserved region, not a broad interior middle). *(WARN item — prevents PASS.)*
- Random-direction control: **yes (n=33)** → **passed**. Norm-matched (‖random.base_dir‖ = ‖S.base_dir‖), independent (per-direction RNG seed), sampled excluding S∪β∪matched. S helix Δ=+0.096 exceeds all 33 (0/33 ≥ S); empirical one-sided p=0.029; z=3.02 above the random-null mean. Capability-selection artifact ruled out (S valid-ORF 0.888 ≈ random-null mean 0.887).
- Sign pattern: preserved (negative α included in M2; no asymmetric flattening).
- Output-case spot-check: not assessable — raw M3 sequence text discarded after metrics (same as before; the C2 pydssp variant retains sequences for its 10 cells).
- Evidence: `code/mechanism.py:22-44`; `code/m3_random_control.py` (33-direction norm-matched null); `results/m2_dose_response_curve.json` (per-dose helix + valid_orf + pLDDT); `results/m3_specificity_summary_v2.json` (locked α=8 deltas + random null: p=0.029, z=3.02).
- Verdict reason: the prior FAIL (decisive stat at a capability-crashed dose + no random-direction control) is fixed; the specificity now rests on a capability-preserved decisive dose with a genuine 33-direction norm-matched null that S beats (p=0.029, z=3.0). Remaining WARN: α not in σ_proj units, positive sweep spans <3 orders of magnitude, and locked α=8 sits at the edge of the capability-preserved region rather than a demonstrated interior plateau middle.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items (WARN — non-blocking, for future polish)
- Express α in σ_proj units (k·σ_proj at the injection site) so doses are comparable across sites/runs/papers.
- Add one or two intermediate doses between α=4 and α=8 (e.g. α=6) and between α=8 and α=16 to demonstrate an interior plateau middle rather than locking at the capability-preserved edge.
- Retain raw generated sequence text for the M3 specificity arms to enable output-case spot-checks (as the C2 pydssp variant already does).
