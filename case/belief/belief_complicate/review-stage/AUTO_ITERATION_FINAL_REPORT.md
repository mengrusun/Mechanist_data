# Auto Iteration Final Report — Belief Localization in Pretrained Pythia LMs

- **Generated**: 2026-07-10T16:20:00
- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 1 / 2
- **Final reviewer score**: 8 / 10
- **Final canonical verdict**: ready
- **Termination reason**: positive_verdict (three-dimensional STOP fired at iteration 2)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The autonomous review loop closed the belief-localization pipeline in a single back-edge action. The reviewer's iteration-1 assessment (score 7 / verdict "almost") identified one blocker — C2 was originally framed as a family-wide Fisher-mask localization claim, but the pythia-410m swap variant produced an integrity-clean negative (Fisher effect 0.3855 inside the 20-random-head 2σ upper bound 0.4675; 2 of 20 random K=20 sets equal or exceeded). Rather than falsely re-labeling the negative as a bug or running new experiments, iteration 1 executed a **type-③ lightweight in-loop rewrite**: C2 was narrowed to C2_v2 with explicit scope {pythia-1b, pythia-2.8b}, and the pythia-410m result was preserved verbatim in the paper as evidence that Fisher-vs-random specificity is scale-emergent (a positive scientific finding, not a robustness failure). The rewrite consumed 1 claim-reentry sub-budget and 0 GPU-hours (the narrowed claim's evidence is already on disk at the two in-scope models). Iteration 2 was a pure reviewer confirmation call: score jumped to 8/10, verdict flipped to "ready," and the three-dimensional STOP fired. C1, C2_v2, C3, C4 all end in INTEGRITY_ONLY (audit-passed, swap-test unresolved by K=1 cap; C1 and C4 additionally have infeasible swap axes with the current Pythia model pool; C3's pythia-2.8b intermediate-checkpoint swap remains the strongest recommended follow-up round).

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 1 | 0 still FAIL; 1 superseded by new claim via ③ (C2 → C2_v2, INTEGRITY_ONLY) |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY (original)| 3 | 3 INTEGRITY_ONLY (unchanged; C1, C3, C4) |
| INTEGRITY_ONLY (produced)| 1 | 1 INTEGRITY_ONLY (C2_v2 — swap-axis infeasible under narrowed scope) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

No claims arrived at the iteration loop in `verify_passed` state. Section skipped.

---

## Section 2 — FAIL Claims (full journey)

### 2.1 `C2` — Belief-Heads Localization (original, family-wide scope)

**Original FAIL signal**
- robustness = 0.00 / 1.00 (threshold = 0.50)
- Inconsistent dimensions: model-swap axis to pythia-410m (PB only; AB ineligible at that scale)
- Variant integrity at entry: **all clean** (Phase-2 main-experiment audit PASS + Phase-9 variant audit PASS = the swap variant faithfully reproduced the main-experiment methodology and returned a real negative for the family-wide scope)

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ③ (lightweight in-loop) | family-wide C2 wording overclaims; 410m negative is a real scale-emergent finding, not a bug | narrowed claim scope to {pythia-1b, pythia-2.8b}; new claim id `C2_v2`; no /auto-experiment run (existing results at both in-scope models already meet all 4 thresholds decisively — Fisher-vs-random gaps 1.7×–86× on random-head 2σ; 19.9×–>1000× on random-mask 2σ); no /auto-verify swap re-run (pythia-410m out-of-scope for C2_v2 by construction; no other Pythia model exists at the model path) | C2_v2 terminal state = INTEGRITY_ONLY (Phase-2 integrity inherited PASS from original C2 audit; swap axis infeasible under narrowed scope) |
| 2 | ⓪ | reviewer confirmation | no back-edge action | reviewer confirms narrowing correctly preserves the scientific content and appropriately reframes the pythia-410m result; STOP fires (score 8, verdict ready) |

**Path taken (summary)**: single type-③ lightweight in-loop rewrite (iteration 1) — claim-reentry sub-budget used: 1

**Experiment & script modifications** (cumulative across all iterations on this claim)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | *no script or plan modifications* | — | — |
| 1 | (rewrite target = claim wording; existing pipeline unchanged) | family-wide C2 claim | scope-narrowed C2_v2 claim (paper-side) |

**Claim modifications**
- **Original claim id** `C2`: "For each Claim-1-eligible Pythia model, distinct HeadSet_personal and HeadSet_attributed exist under Fisher masks (top-0.1% F_target ∧ ¬ top-1% F_knowledge) and zero-ablation of each minimal head set satisfies target-drop ≥ 0.30, effect outside the 20-random-head 2σ band, off-target drops ≤ 0.10 (both other belief and world_knowledge), and Pile PPL ≤ clean × 1.05."
- **After iteration 1 → new claim id** `C2_v2`: "**At pythia-1b and pythia-2.8b** (the Claim-1-eligible Pythia scales), distinct HeadSet_personal and HeadSet_attributed exist under Fisher masks (top-0.1% F_target ∧ ¬ top-1% F_knowledge) and zero-ablation of each minimal head set satisfies target-drop ≥ 0.30, effect outside the 20-random-head 2σ band, off-target drops ≤ 0.10 (both other belief and world_knowledge), and Pile PPL ≤ clean × 1.05. **Fisher-vs-random specificity is scale-emergent**: at pythia-410m the K=20 Fisher-selected head set produces a substantial behavioral drop (tgt_drop = 0.3855 ≥ 0.30) but is not statistically distinguishable from random K=20 head sets at the 2σ level (Fisher effect 0.3855 falls inside the random-head 2σ upper bound 0.4675; 2 of 20 random sets equal-or-exceed), indicating diffuse / redundant PB processing at that scale. This scale-dependence is a positive scientific finding, consistent with C1's observation that AB is at chance at pythia-410m."
- **Scope change**: (a) restrict claim scope to the Claim-1-eligible models (pythia-1b, pythia-2.8b); (b) preserve the pythia-410m result as evidence that Fisher-vs-random specificity is *scale-emergent*, not scale-invariant; (c) explicitly link the 410m diffuse-circuit result to C1's finding that pythia-410m fails the AB above-chance eligibility criterion.

**Final experiment summary**
- New runs cited: none (0 new `/run-experiment` calls; the narrowed claim's evidence is entirely on disk at `runs/M2a_pythia-1b_*/`, `runs/M2b_pythia-1b_*/`, `runs/M2c_pythia-1b_*/`, `runs/M2d_pythia-1b_*/`, `runs/M2e_pythia-1b_*/`, and the same paths for pythia-2.8b)
- Final robustness: — (INTEGRITY_ONLY under narrowed scope; no in-scope swap variant available with the current Pythia model pool)
- Final variant pass rate: — (swap axis infeasible under narrowed scope)
- Final status: **superseded by C2_v2 = INTEGRITY_ONLY** — the original C2 (family-wide) is not the tracked claim; C2_v2 (narrowed) is Open-Items INTEGRITY_ONLY

**Reviewer memory thread** (cross-iteration suspicions filtered to this claim's pattern)
- Iter 1 "C2 overclaim risk" → Iter 2: **addressed** (narrowing correctly reframes 410m as scale-emergent)
- Iter 1 "410m interpretation nuance" → Iter 2: **addressed** (wording explicitly separates behaviorally effective Fisher ablation from lack of statistical uniqueness vs random sets)
- Iter 1 "off-target AB inversion at 410m" → Iter 2: **carried forward as cautious exploratory observation only** — not built into the main claim, kept as a minor exploratory remark in the paper because AB baseline at pythia-410m is at chance (0.46) so the +0.308 inversion signal is fragile

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

No claims arrived at the iteration loop in `verify_inconclusive` state. Section skipped.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

No claims arrived at the iteration loop in `verify_zero_eligible_variants` state. Section skipped.

### 4b — INTEGRITY_ONLY Claims (no-action; carried to Open Items)

These claims did not require any iteration-loop action. Their Phase-2 audits passed and their Stage-2 swap tests were intentionally skipped (K=1 cap) or independently infeasible with the current Pythia model pool. All four are surfaced in Section 8.

- **C1 — Scale-Dependent Emergence**: audit passed; Stage-2 skipped by cap; model-swap independently infeasible (all 3 available Pythia models already in the main experiment). Upgrade command: `/auto-verify C1 --resume: true`.
- **C2_v2 — Belief-Heads Localization (narrowed to pythia-1b/2.8b)**: audit inherited PASS from original C2; swap axis infeasible under narrowed scope (pythia-410m out-of-scope; no other Pythia model at model path). Upgrade path requires provisioning a Pythia checkpoint outside {pythia-1b, pythia-2.8b} that also has above-chance AB.
- **C3 — Formation Window**: audit passed; Stage-2 skipped by cap. Feasible follow-up: pythia-2.8b intermediate checkpoints ARE available at `/mnt/quarkfs/share_model/Ptyhia/pythia-2.8b-checkpoints/` and pythia-2.8b Claim-2 head sets are on disk. **Recommended strongest follow-up round**. Upgrade command: `/auto-verify C3 --resume: true`.
- **C4 — Dynamic Controllability**: audit passed; Stage-2 skipped by cap; model-swap to pythia-410m infeasible because AB at chance at pythia-410m → no valid AB head set → router non-functional. Upgrade command: `/auto-verify C4 --resume: true`.

---

## Section 5 — Legacy DEFERRED Claims (empty)

No legacy `deferred_claims` bucket in new /auto-verify runs. Section retained per template for backward compatibility only; empty here.

---

## Section 6 — Cross-Cutting Patterns

Patterns the reviewer flagged across both iterations (deduped, verbatim source `REVIEWER_MEMORY.md`):

- **Scale-dependent circuit concentration**: The Fisher-vs-random gap widens dramatically with scale (1.7× → 86× for PB; 6.7× → 35× for AB across pythia-1b → pythia-2.8b). The C2_v2 rewrite foregrounds this as the core mechanistic finding. **Touched claims**: C2, C2_v2, C1. **Resolved at termination**: yes — the paper narrative now presents scale-dependence as a positive finding, and the reviewer explicitly commended the reframing.
- **AB emerges sharply, PB matures gradually**: behavioral (C1) and formation-window (C3) evidence converge on "AB step-function at 1B, PB non-monotonic across scale + gradual across pretraining." **Touched claims**: C1, C3. **Resolved at termination**: yes — the pattern is coherent across the two claims and does not require further work.
- **Off-target inversion at 410m** (AB accuracy improves by 0.308 abs when PB Fisher heads are ablated at pythia-410m): possible cross-frame interference / inhibitory coupling at smaller model sizes. **Touched claims**: C2 → C2_v2. **Resolved at termination**: partially — kept as a cautious exploratory observation only; not built into the main claim because AB baseline at pythia-410m is at chance so the signal is fragile.
- **C4 pre-head classifier is a frame router, not a deep-belief probe**: L*=1 classifier likely leverages distinguishing prompt tokens ("believes", "thinks", "In reality") — a design consequence of task.md's fixed-prompt constraint, not a bug. **Touched claims**: C4. **Resolved at termination**: not fully — remains as a paper-side wording discipline check ("frame router" language should be used prominently). No further loop action needed; the underlying result stands.
- **Reproduction faithfulness, no goalpost-moving**: reviewer explicitly commended that the authors respected the fixed C2 thresholds (target_drop ≥ 0.30, outside 20-random-head 2σ, off-target ≤ 0.10, Pile PPL ≤ 1.05×) and did not weaken them to force a PASS at pythia-410m. **Touched claims**: C2. **Resolved**: yes.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 1 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ③ claim_reentry (lightweight_in_loop) | C2 | C2_v2 | 0 | 0.0 | 7 | almost |
| 2 | ⓪ (pure review; no back-edge — does not consume iteration) | C2_v2 (confirmation) | — | 0 | 0.0 | 8 | ready |

---

## Section 8 — Open Items for Human Reviewer

Items the loop could not close within its scope. All are either recorded caveats for the paper or recommended follow-up rounds; none block submission per the STOP contract.

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - `C1` [stage2_skip_reason: max_verify_claims_cap]:
    * Upgrade: `/auto-verify C1 --resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
    * **Additional constraint**: model-swap independently infeasible — all 3 available Pythia final checkpoints (pythia-410m, pythia-1b, pythia-2.8b) are already in the main experiment. Requires provisioning an additional Pythia checkpoint (e.g., pythia-70m, pythia-160m, or pythia-6.9b) at `/mnt/quarkfs/share_model/Ptyhia/`.
    * main-experiment integrity: pass
  - `C2_v2` [stage2_skip_reason: max_verify_claims_cap AND model-swap infeasible under narrowed scope]:
    * Upgrade path: adding a Pythia checkpoint outside {pythia-1b, pythia-2.8b} that also has above-chance AB in-domain would enable a fresh swap-test. Under the current model pool the swap axis is exhausted.
    * main-experiment integrity: pass (inherited from original C2 Phase-2 audit — the narrowing did not change any script)
  - `C3` [stage2_skip_reason: max_verify_claims_cap]:
    * Upgrade: `/auto-verify C3 --resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
    * **Actionable — strongest recommended follow-up round**: pythia-2.8b intermediate checkpoints ARE available at `/mnt/quarkfs/share_model/Ptyhia/pythia-2.8b-checkpoints/` and pythia-2.8b Claim-2 head sets exist at `runs/M2c_pythia-2.8b_*/headset.json` — a model-swap variant would replay M3.a behavioral + M3.b causal trajectories at pythia-2.8b.
    * main-experiment integrity: pass
  - `C4` [stage2_skip_reason: max_verify_claims_cap]:
    * Upgrade: `/auto-verify C4 --resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
    * **Additional constraint**: model-swap to pythia-410m is infeasible because pythia-410m AB is at chance (acc=0.460, p=0.96), so no valid AB head set can be identified at that scale → the frame-classifier-driven amplifier requires both PB and AB head sets to function.
    * main-experiment integrity: pass
- **Legacy deferred claims**: none (bucket empty in new /auto-verify runs).
- **Recurring unresolved patterns** (paper-side wording discipline checks — do NOT block submission):
  - **C4 wording**: consistently present the pre-head classifier as a **frame router under fixed prompts**, NOT as strong evidence of a deep latent belief-state decoding.
  - **pythia-410m AB inversion**: keep as a cautious exploratory observation only; do not overinterpret; AB baseline at pythia-410m is at chance so the +0.308 inversion signal is fragile.
  - **C1 scope**: stay descriptive within the tested Pythia checkpoints; avoid broad developmental-law language.
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none — sub-budget was 1/2 used, no refusals needed.
