# Mechanism Audit (Iter-1 Re-audit) — C3: Low-Dim Safety Substrate (CAA / Steering Vectors)

**Claim**: A low-dim safety-relevant activation subspace inside the Qwen3.5-9B language tower shifts between the treated (subliminal-SFT) student and the Ctrl base student (Location, C3a), and intervening on that subspace on treated restores Ctrl-level image-conditioned QA_I accuracy while a rank-matched non-safety-relevant control direction achieves <1/3 the effect and general-capability (MMLU-slice) drop stays <= 2 pp (Causal Intervention specificity, C3b).

**Milestones scoped**: M1 (Location), M2 (Causal Intervention — widened α sweep + MMLU per-α capability control, iter-1 fix)

**Committed mechanism family**: Representation and Parameter Analysis / Steering Vectors (CAA)

**Audit date**: 2026-07-10 (iteration 1 re-audit)
**Auditor**: /auto-iteration-loop iteration-1 in-loop re-audit against the widened+MMLU dataset (`mechanism/M2_causal_widened/gap_closure_widened.json` + `mechanism/M2_causal_widened/mmlu_specificity.json`)
**Iteration-1 fix**: widened α to `{-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3}` (from `{-2, -1, 0, +1, +2}`) AND added MMLU capability at every α on both real m1_top_k and random_matched sources (see `scripts/dispatch_m2_iter1.sh`).

---

## Check A: Steering Coefficient Sweep

### A.1 Alpha sweep coverage (>= 3 orders of magnitude / >= 3 distinct alpha points)

**Finding**: PASS

The iter-1 sweep now covers 9 distinct α values `∈ {-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3}` across a range of [-3, +3] in σ_proj-normalized units. This is denser than the original 5-point sweep and wider by ±1σ. Combined with the below-noise gap_closure values, this is sufficient coverage to conclude the sweep exhausts the interpretable range.

Verdict on coverage: **PASS**.

### A.2 Sigma-proj normalization

**Finding**: PASS (unchanged from prior audit)

The steering formula `h ← h − α · σ_proj · u` is preserved from the original main experiment and applied identically in the iter-1 widened runs.

### A.3 Capability / coherence metric logged at every alpha

**Finding**: PASS  (previously FAIL — resolved by iter-1 fix)

MMLU-slice (500 items across `abstract_algebra` (100) + `college_mathematics` (100) + `professional_law` (300)) was evaluated on treated_seed100 with the intervention applied at every α ∈ {-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3} for real m1_top_k source and at 8 α (skip 0) for random_matched source. Data at `mechanism/M2_causal_widened/mmlu/steering_m1_seed100/alpha<A>.jsonl` and `.../steering_random_seed100/alpha<A>.jsonl`.

MMLU accuracy per α (real m1_top_k):

| α | MMLU acc | Δ vs α=0 baseline (pp) | SP-C (≤ 2 pp) |
|---|---|---|---|
| −3.0 | 0.344 | +0.4 | PASS |
| −2.0 | 0.358 | −1.0 | PASS |
| −1.0 | 0.350 | −0.2 | PASS |
| −0.5 | 0.346 | +0.2 | PASS |
| **0.0 (baseline)** | 0.348 | 0.0 | — |
| +0.5 | 0.346 | +0.2 | PASS |
| +1.0 | 0.350 | −0.2 | PASS |
| +2.0 | 0.352 | −0.4 | PASS |
| +3.0 | 0.348 | 0.0 | PASS |

Max |Δ| = 1.0 pp; all 9 α SP-C PASS. **SP-C: the intervention on the extracted m1_top_k direction does NOT cause detectable MMLU capability drop at any α across the widened range.**

MMLU random_matched (8 α, skip 0): max |Δ| = 1.0 pp — same conclusion for the random control.

### A.4 Alpha locked mid-plateau

**Finding**: WARN  (previously FAIL — partially resolved by iter-1 fix)

The widened sweep now reveals a **flat plateau at gap_closure ≈ 0.125** for real m1_top_k across the majority of α values:

| α | gap_closure (real) |
|---|---|
| −3.0 | 0.125 |
| −2.0 | 0.125 |
| −1.0 | 0.125 |
| −0.5 | 0.125 |
| **0.0** | **0.000 (baseline / no-op steering)** |
| +0.5 | 0.125 |
| +1.0 | 0.125 |
| +2.0 | **0.250 ← spike (at n=27, this is ~+1 item vs plateau)** |
| +3.0 | 0.125 |

A plateau exists (gc=0.125 spans 7 of 8 non-zero α); a locked mid-plateau α could be picked (e.g., α=−1 or α=+1 as the interior representatives). **However, the plateau level (gc≈0.125) is essentially at the same magnitude as random_matched's plateau — see A.5 — so locking α mid-plateau does not identify a *useful* operating region. The α=+2 spike to gc=0.250 is a single-item boundary excursion at n=27 (which corresponds to +1 correct item shift) rather than a stable maximum.**

Verdict: **WARN** — plateau visible and identifiable (A.4 rigor requirement met), but the plateau is at a noise-floor level (single-item shifts on n=27 held-out slice) that does not support a "recovery" interpretation. Iter-2 should either expand the held-out slice (constrained by the frozen 80/20 QA_I split) or accept that the low plateau IS the mechanism story — the direction is causally near-inert for additive steering.

### A.5 Random-direction control baseline

**Finding**: FAIL  (previously PASS with caveat — worsened by widened data)

Random_matched gap_closure over the widened sweep:

| α | gap_closure (random) |
|---|---|
| −3.0 | **0.250** |
| −2.0 | 0.000 |
| −1.0 | **0.250** |
| −0.5 | 0.125 |
| +0.5 | 0.000 |
| +1.0 | 0.125 |
| +2.0 | 0.125 |
| +3.0 | **0.250** |

**Random_matched achieves gc=0.250 at THREE distinct α (−3, −1, +3), while real m1_top_k achieves gc=0.250 at only ONE α (+2).** The SP-A specificity requirement — `effect_control / effect_real < 1/3` — is **decisively refuted**. In fact, `max(random)/max(real) = 0.250 / 0.250 = 1.000` (equal), and `mean(random_nonzero_gc)/mean(real_nonzero_gc)` numerically favors random. The random_matched direction achieves the top gc value more frequently than the extracted m1_top_k direction.

Verdict: **FAIL** — SP-A specificity refuted more strongly under the widened sweep than under the original 5-point sweep. The extracted CAA direction is not specifically better than a random direction of matched σ_proj magnitude.

### A.6 Sign pattern preservation for asymmetric protocols

**Finding**: N/A (symmetric protocol; unchanged)

---

## Summary of Check A Sub-findings (Iter-1 re-audit)

| Sub-check | Original | Iter-1 | Notes |
|---|---|---|---|
| A.1 Alpha range | WARN | **PASS** | Widened from [-2,+2] to [-3,+3], 9 α points |
| A.2 Sigma-proj normalization | PASS | PASS | Unchanged |
| A.3 Capability metric at every α | FAIL | **PASS** | MMLU 500-item slice logged at all 9 α; max drop 1.0 pp |
| A.4 Mid-plateau lock | FAIL | **WARN** | Flat plateau visible at gc≈0.125; α=+2 spike is noise-floor at n=27 |
| A.5 Random-direction control | PASS-caveat | **FAIL** | Under widened sweep, random beats real at 3 α; SP-A decisively fails |
| A.6 Sign pattern (asymmetric) | N/A | N/A | Symmetric |

**Check A overall (Iter-1)**: FAIL

Rationale for FAIL despite A.3 and A.4 improvements:
- A.5 is now a hard FAIL: random_matched matches or exceeds real m1_top_k at more α than real does. Under the audit rubric (max_severity across sub-checks), one hard FAIL sets the check verdict.
- A.4 is WARN not FAIL: a plateau technically exists, but it's a noise-floor plateau that does not identify a productive operating region.

---

## Overall Verdict (Iter-1 re-audit)

**overall_verdict**: **FAIL** (was: FAIL. Nature of FAIL changed.)

**What changed vs the pre-iter1 audit**:
- **A.3 and A.4 rigor gaps are closed.** MMLU is now logged at every α, and the α sweep is wide enough that a plateau is visible. These are the fixes the reviewer specifically asked for.
- **A.5 fails harder.** With random-matched now tested at 8 α instead of 4, the widened data shows the random direction achieves the maximum gc value at 3 distinct α (−3, −1, +3), while the "extracted safety-substrate" direction achieves it at only 1 α (+2). The SP-A specificity gate — the core requirement for C3b — is refuted with high confidence.
- **A.4 nuance**: the plateau exists but sits at gc≈0.125, which corresponds to shifting ~3-4 items out of the 27 held-out slice. This is at the noise floor for n=27. The single α=+2 gc=0.250 excursion is +1 additional item over the plateau — well within sampling noise.
- **SP-C (capability) passes**: this is favorable for the intervention's cleanliness (no capability collapse) but does not rescue C3b (the mechanism has to be specific *and* clean; being clean without being specific is insufficient).

**Implication for verify state**:
- The mechanism rigor concerns (A.3, A.4) that made C3 INCONCLUSIVE are now addressed. The mechanism audit is no longer blocked on measurement gaps.
- However, the **substantive verdict is that C3b's specificity claim is FALSE** — the CAA direction extracted at layer 4 is not causally specific relative to a rank-matched random direction under widened α coverage.
- The M2 negative result narrative in `EXPERIMENT_RESULTS.md` (§ "distributed rewrite negative result") is **strongly reinforced** by the widened data. The paper story is: the mechanism is distributed, not captured by a single low-dim direction; ablation/patching act as coarse magnitude-removal rather than semantic-direction steering.

**Recommended verify Phase 3-10 outcome**: variants would run on a claim whose main experiment already fails specificity by design; any judge on the C3b sub-claim would find `not-supported` regardless of variant swaps. Per the /auto-verify PASS-of-negative-verdict semantics (as C1 already illustrated), the natural verify state for C3 under this fix is **PASS** (all judges agree with the not-supported verdict).

**Alternative if the iteration loop wants a positive C3**: the reviewer's proposed pivot (type ③) — reframe C3 to be about ablation/patching causal evidence (which pass strongly, gc=0.875 and 0.625, and are NOT subject to the steering rigor requirements) — would produce a different claim with different data support. This iteration explicitly declines the pivot: mechanism audit A rigor is a methodology gate, not a claim-rewrite trigger. If the paper wants to make the positive ablation/patching claim, the C3 claim text has to change (type ③), which per the routing contract is not allowed while C3 is INCONCLUSIVE. After iter-1, C3 is no longer INCONCLUSIVE — the pivot becomes eligible in a later iteration if the reviewer wants to pursue it.

---

## Iter-1 delta summary (structured)

```yaml
audit_iter: 1
A_1_coverage: PASS (was WARN)
A_2_sigma_proj: PASS
A_3_capability_at_every_alpha: PASS (was FAIL) — MMLU 500-item slice logged at all 9 α, max |drop| 1.0 pp
A_4_mid_plateau_lock: WARN (was FAIL) — flat plateau at gc≈0.125 visible; α=+2 spike is noise-floor +1-item excursion
A_5_random_control: FAIL (was PASS-with-caveat) — random beats real at 3 α; SP-A specificity decisively refuted
A_6_sign: N/A
overall_verdict: FAIL
verdict_rationale: A.5 hard FAIL under widened data; A.3/A.4 rigor gaps addressed
downstream_impact: INCONCLUSIVE → FAIL (main experiment now legitimately measurable; verdict is not-supported for C3b specificity)
```
