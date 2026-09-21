# Auto Iteration Final Report — Verbal-Confidence Cache (C1)

- **Generated**: 2026-07-14T16:00:00Z
- **Iterations consumed**: 4 / 6 (plus 1 type-⓪ narrative-closure iteration that does not count against budget)
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: **7 / 10**
- **Final canonical verdict**: **ready**
- **Termination reason**: `positive_verdict` — score ≥ TARGET_SCORE=6 AND verdict ∈ {ready, almost}. **Caveat**: condition 3 of the three-dimensional STOP rule is NOT formally met on-disk — C1 remains `verify_inconclusive` in `VERIFY_REPORT.md` because the on-disk `/auto-verify` audit has not been re-run. The iteration-loop external reviewer (gpt-5.4) has explicitly certified the mechanism-audit gate as PASSING, and recommends the orchestrator invoke `/auto-verify C1` as a post-loop step to formally close the state.
- **Cumulative cost**: runs_total=4, gpu_hours_total=2.287 (well within the 10-h task.md budget; main experiment ≈ 3.7 h + iteration 2.29 h ≈ 6.0 h total)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

Over 4 back-edge iterations (all type ②, no claim rewrites) and 1 final narrative-closure iteration (type ⓪), the loop transformed C1's `verify_inconclusive` state (Phase 2 combined FAIL: exp=WARN, mech=FAIL) into a reviewer-certified **READY** verdict at score 7/10.

**What happened**: the mechanism-audit's five failure points on M5 (no capability metric, no locked α, no random-direction control, no raw text log, effect within noise floor) were all repaired in iteration 1's M5-v2 script (`scripts/m5_steer_v2.py`); the experiment-audit's four WARN points (P2 aggregation opacity, `answer_acc_preserved` trivially true, P5 cross-seed inconsistency, seed2024 pending) were addressed by a hardened aggregator (`scripts/aggregate_results_v2.py`) and the completion of seed2024. Iterations 2, 3, 4 progressively closed reviewer-flagged loopholes: sub-argmax (M5-v3), site-selection narrowness (top-5-site sweep), and distributed multi-site causal use (joint 5-site sweep).

**Final scientific position**: C1's headline strong causal cache-and-retrieve claim is NOT supported by the evidence. The weaker DECODABILITY–CAUSAL-CONTROL DISSOCIATION claim IS supported at three levels (argmax null, logit null, joint null with a small ~0.32-unit distributed signal). The paper is submittable as a bounded negative-result / dissociation paper. The reviewer explicitly discouraged overreaching phrasing like "confidence is not causally used" or "causal null" — the correct framing is "decodable but not strongly controllable under tested interventions".

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 1 | 1 mechanism-audit gate CERTIFIED-PASS by reviewer at score 7/10 verdict ready; on-disk state still `verify_inconclusive` pending orchestrator `/auto-verify` re-run |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY (legacy)  | 0 | — |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

*(none — the loop entered with 0 verify_passed claims)*

---

## Section 2 — FAIL Claims (full journey)

*(none — the loop entered with 0 verify_failed claims)*

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

### 3.1 `C1` — "Post-answer hidden states carry a retrievable representation of the model's self-assessed verbal-confidence score; the score is CACHED at answer-emission positions (E0..E4, layers 5-10) rather than computed on-demand at the confidence-generation position (C0)."

**Original INCONCLUSIVE reason** (from `verify/C1_verbal_confidence_cache/ROBUSTNESS.md`'s `inconclusive_reason`):
- Main-experiment integrity broken (Phase 2 combined FAIL: exp=WARN, mech=FAIL). Mechanism audit FAIL was the driver.
- Mech-FAIL details: M5 has no independent capability metric; target effect (span ≈ 1.5 verbal-conf units) is within baseline-noise floor (per-item std ≈ 41); no locked α; no random-direction control; no raw text samples.
- Exp-WARN details: P2 ratio aggregation not derivable from per-seed values; `answer_acc_preserved=1.0` trivially true by construction; P5 cross-seed inconsistency (seed42=3.57, seed123=0.10 with seed2024 still running).

**Experiment plan & script modifications** (cumulative across all iterations on this claim)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `refine-logs/EXPERIMENT_PLAN.md` M5 section | v1 grid α ∈ [-4,-2,-1,0,1,2,4], method ∈ [diff_of_means, lda], success = "monotone R² AND large effect at ±4 AND answer_acc_preserved" (last was vacuous by construction). | v1 kept for reference; v2 declared AUTHORITATIVE. v2 grid extended to α ∈ [-16,-8,-4,-1,0,1,4,8,16]; adds `n_random_directions=10`, `capability_probe=teacher-forced NLL on unrelated continuation`, `capability_tol_nats=0.3`, `n_sample_texts_per_alpha=5`. New success = `|conf_effect_at_alpha_star| > 5.0` AND `trained_beats_random_at_alpha_star == true` AND `capability_preserved_at_alpha_star == true`. `resource_fidelity: cost-aware` marker preserved verbatim. |
| 1 | New script `scripts/m5_steer_v2.py` | — | 400 lines: extended α, capability probe (teacher-forced NLL on "The quick brown fox jumps over the lazy dog while the sun sets."), n=8 random-direction control, locked α*, greedy 20-token text samples for 5 items per α. |
| 1 | New script `scripts/run_m5_v2_all_seeds.py` | — | Launcher that reuses one loaded model across seeds (saves ~10 min per skipped model load). |
| 1 | New script `scripts/aggregate_results_v2.py` | — | Hardened aggregator: per-seed P2 ratio transparency; explicit "vacuous by construction" note on `answer_acc_preserved`; P5 M6c variance gate (HOLD when values span > 2 units AND min < 0.5 AND max > 2.0); v2 P4 verdict gates. Overwrites `refine-logs/EXPERIMENT_RESULTS.md` with v2 report and `results/all_summary_v2.json`. |
| 2 | New script `scripts/m5_steer_v3_logit.py` | — | Logit-level supplement: reads softmax over 10 digit tokens at C0 under intervention; computes `E[first_digit] = Σ d · P(digit=d)` and digit-entropy `H_digit`. Closes sub-argmax loophole. |
| 3 | `scripts/m5_steer_v3_logit.py` extension | Single-site only. | Added `--sites` comma-list arg; added `--out_suffix` tag. Iteration-2 outputs renamed to `m5v3_expected_score_E4L10_seed{S}.json`. |
| 3 | `scripts/aggregate_results_v2.py` extension | Reads only E4L10 v3 output. | Reads per-site rollup for top-5 sites; adds `site_sweep_verdict` = "dissociation_generalizes" iff max |span| across ALL 5 sites and 3 seeds < 0.5. |
| 4 | New script `scripts/m5_steer_v3_joint.py` | — | `MultiSiteSteeringHook` class: one PyTorch forward hook per unique layer, each firing on ALL of that layer's sites simultaneously with per-site (patch_pos, direction, α·σ_proj_s) triples. Reduced α grid to {-16,-8,-4,0,4,8,16}. |
| 4 | `scripts/aggregate_results_v2.py` extension | No joint. | Adds m5_v3_joint rollup + `joint_verdict` = "distributed_null" iff |span| max < 0.5. |

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ② | mechanism-audit FAIL + experiment-audit WARN (see original INCONCLUSIVE reason) | Wrote M5-v2 hardened script + hardened aggregator; ran 3 seeds × diff_of_means × 9 α × 8 random dirs (1.665 GPU-h on GPU 2) | Score 3→3, verdict still not_ready. M5-v2 shows greedy conf BIT-IDENTICAL across α ∈ [-16, +16] for both trained and 8 random dirs on seeds 42/123. Capability preserved (NLL delta < 0.02 nats/token). Definitive argmax null at E4L10. |
| 2 | ② | sub-argmax escape hatch flagged | Wrote M5-v3 logit-level script; ran 3 seeds × 40 items × 9 α at E4L10 (0.088 GPU-h) | Score 3→4, verdict still not_ready. Digit-distribution E[first_digit] span across α ∈ [-16, +16]: seed42=-0.009, seed123=-0.003, seed2024=-0.009 (all << 0.5 threshold). Sub-argmax null CONFIRMED. |
| 3 | ② | site-selection narrowness — is E4L10-null site-specific? | Extended M5-v3 script for multi-site; ran top-4 additional sites × 3 seeds (0.347 GPU-h) | Score 4→5, verdict still not_ready. Top-5 site spans (max |span|): E4L10=0.009, E1L5=0.146, E2L5=0.047, E3L10=0.016, E3L5=0.009. All < 0.5. `dissociation_generalizes` verdict. |
| 4 | ② | distributed multi-site causal use loophole — single-site nulls may miss joint control | Wrote M5-v3-joint script with multi-hook simultaneous perturbation; ran 3 seeds × 40 items × 7 α × 5-site joint (0.187 GPU-h) | Score 5→6, verdict "almost". Joint spans: seed42=0.320, seed123=0.140, seed2024=0.200. `distributed_null` verdict — max |span|=0.32 < 0.5 threshold — with a small distributed signal detected that single-site sweeps missed. |
| 5 | ⓪ | none — reviewer honored iteration-4 commitment ("if joint intervention comes back near-null, 7/10 ready") | Narrative closure — no experiments | Score 6→7, verdict "ready". Mechanism-audit gate certified as PASS for narrowed dissociation claim. Recommended paper framing and title. |

**Claim modifications** (mandatory subsection — write `none — variant fix path only` if no rewrite happened)
- Original claim id `C1`: "Post-answer hidden states carry a retrievable representation of the model's self-assessed verbal-confidence score; the score is CACHED at answer-emission positions (E0..E4, layers 5-10) rather than computed on-demand at the confidence-generation position (C0)."
- After iteration 5 → **no rewrite**. C1 wording retained as-is because rewrite is not available for INCONCLUSIVE claims per the routing contract.
- **Paper framing recommended by reviewer (for the write-up, not a code-level claim change)**:
  - Title: *"Decodable but Not Strongly Controllable: A Dissociation Between Confidence Readout and Single-Site Causal Steering in Gemma-3-27B"*
  - The paper should say: "confidence is linearly decodable from post-answer residuals in Gemma-3-27B-pt on TriviaQA rc.nocontext, but direct linear steering at the strongest decodable sites — individually or jointly across the top-5 — does not produce strong causal control of the confidence readout".
  - The paper should NOT say: "confidence is not causally used" / "causal null" / "epiphenomenal representation".
  - The paper should honestly note: joint top-5 shows a small distributed effect (~0.32 first-digit units, ≈3.2 verbal-conf-score points) that single-site sweeps missed — this keeps the door open for "weak distributed causal sensitivity without strong steering leverage".

**Re-experiment outcome**

| Iter | Path | New runs | Result |
|---|---|---|---|
| 1 | Wrote M5-v2 script + launcher + aggregator v2. | `runs/iteration_round_1/m5v2_full/` (3 seeds × diff_of_means × 9 α × 8 rand dirs, 1.665 GPU-h) | argmax null at E4L10 across 3 seeds |
| 2 | Wrote M5-v3 script. | `runs/iteration_round_2/m5v3_logit/` (3 seeds × 40 items × 9 α, 0.088 GPU-h) | logit-level (sub-argmax) null at E4L10 across 3 seeds |
| 3 | Extended M5-v3 for multi-site. | `runs/iteration_round_3/m5v3_sitesweep/` (4 sites × 3 seeds × 40 items × 9 α, 0.347 GPU-h) | site-generalized null at top-5 sites |
| 4 | Wrote M5-v3-joint script. | `runs/iteration_round_4/m5v3_joint/` (5-site simultaneous × 3 seeds × 40 items × 7 α, 0.187 GPU-h) | distributed_null with small joint effect detected |

**Final status**: **RECOMMENDED-PASS (for narrowed dissociation claim)** — mechanism-audit gate certified by iteration-loop reviewer; formal PASS pending orchestrator `/auto-verify` re-run.

**Reviewer memory thread**
- Iter 1: probe (M2) is only clean positive; M5 non-diagnostic; P2 aggregation opaque; answer_acc_preserved trivially true; M6c unstable.
- Iter 2: M5 non-diagnostic RESOLVED; P2 aggregation RESOLVED; answer_acc_preserved misuse RESOLVED; M6c missing seed2024 RESOLVED. New: sub-argmax loophole.
- Iter 3: sub-argmax loophole RESOLVED. New: site-selection narrowness.
- Iter 4: site-selection narrowness RESOLVED. New: distributed multi-site causal use loophole.
- Iter 5: distributed multi-site RESOLVED (small effect detected but < 0.5 threshold). Final residual: OVERCLAIMING risk in the paper writeup — must frame as "decodable but not strongly controllable under tested interventions", not "not causally used".

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*(none — the loop entered with 0 verify_zero_eligible_variants claims)*

---

## Section 5 — Legacy DEFERRED Claims

> Empty in new runs. Retained for backward compatibility only.

---

## Section 6 — Cross-Cutting Patterns

- **"Passing" predicates trivially or non-diagnostically passing** (first flagged iter 1): resolved by demoting `answer_acc_preserved=1.0` as vacuous-by-construction and adding independent capability metric (M5-v2 unrelated-continuation NLL).
- **Effect sizes at cache site are on the same order as per-item baseline std** (first flagged iter 1): confirmed as a REAL negative finding, not a methodology bug. The M5-v2 direction produces near-zero effect at α up to 16σ_proj while capability preservation NLL delta stays < 0.02 nats/token — the intervention IS applied but has no downstream effect on greedy confidence decode.
- **Aggregation transparency issue: reported ratios don't match derivable per-seed numbers** (first flagged iter 1): resolved iter 1 by aggregator v2's Recipe-A (per-seed ratios) + Recipe-B (global aggregate) reporting.
- **Robust decodability without causal efficacy** (first flagged iter 2): this is the CENTRAL finding of the paper. Confirmed across argmax (v2), logit level (v3), site sweep (v3-sitesweep), and joint intervention (v3-joint).
- **Most "passes" disappear once trivial/non-diagnostic criteria are removed** (first flagged iter 2): confirmed — original v1 report claimed 2/5 predicates pass (P1 + P5); v2 report shows 1/5 with P5 downgraded to HOLD due to variance gate. Only P1 (probing) survives strict methodology.
- **Cross-seed variability suggests fragility, not crisp mechanism** (first flagged iter 2): unresolved — P5 M6c per-seed values [3.57, 0.10, 0.43] retained as HOLD verdict throughout. Reviewer acknowledged as acceptable caveat, not blocking.
- **Overclaiming risk** (first flagged iter 5): main residual concern. Not a methodology issue — a paper-writing concern.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 4 / 6 (plus 1 type-⓪ narrative-closure iteration that does not count against budget)
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 4 (each launched via direct GPU dispatch, one dispatch per iteration; per-iteration cost recorded in per-run `cost.json`)
- **Iteration GPU-hours**: gpu_hours_total = 2.287
- **Cumulative pipeline GPU-hours (main + iteration)**: ≈ 3.7 + 2.29 ≈ 6.0 h (task.md budget: 10 h; remaining: ~4 h)

### Per-iteration breakdown (from `iteration_breakdown[]`)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② plan_script_rerun | C1 | — | 1 | 1.665 | 3 | not ready |
| 2 | ② plan_script_rerun | C1 | — | 1 | 0.088 | 4 | not ready |
| 3 | ② plan_script_rerun | C1 | — | 1 | 0.347 | 5 | not ready |
| 4 | ② plan_script_rerun | C1 | — | 1 | 0.187 | 6 | almost |
| 5 | ⓪ narrative_closure | C1 | — | 0 | 0.000 | 7 | **ready** |

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims** (after exhausting routing options): none
- **Still-INCONCLUSIVE claims**:
  - **C1 (formally-on-disk)** — mechanism-audit gate CERTIFIED-PASS by iteration-loop reviewer at score 7/10 verdict ready; on-disk `VERIFY_REPORT.md` NOT updated by this loop. **Recommended orchestrator action**: invoke `/auto-verify C1 — resume: false, swap_variants: true, dimensions: model` to formally close the state. Expected: Phase 2 mechanism-audit PASS (based on M5-v2 + M5-v3 + joint evidence); Stage 2 model-swap variants (Qwen 2.5 7B per task.md) may run and produce a final PASS or ZERO_ELIGIBLE_VARIANTS verdict. Given the negative causal finding, the swap variants would be checking whether the same null result holds under a smaller model.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**: none
- **Legacy deferred claims**: none
- **Recurring unresolved patterns**:
  - P5 M6c cross-seed variability [3.57, 0.10, 0.43] persists as a caveat. Reviewer certified as acceptable within the narrowed dissociation framing.
  - Overclaiming risk in paper writeup — the paper must be worded as "decodable but not strongly controllable under tested interventions", not "not causally used" or "causal null" or "epiphenomenal".
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none — reviewer never requested ③ (correctly identified as unavailable for INCONCLUSIVE per the routing contract).

---

## Appendix — Key files produced/modified

Under the working directory `/data/zhenqian/Reproduction1/mechanica/belief/verbal_confidence/`:

### New scripts
- `scripts/m5_steer_v2.py` — hardened M5 with capability probe + random-direction control + locked α* + raw text logging.
- `scripts/run_m5_v2_all_seeds.py` — multi-seed launcher for M5-v2.
- `scripts/m5_steer_v3_logit.py` — logit-level supplement + multi-site support.
- `scripts/m5_steer_v3_joint.py` — joint multi-site simultaneous steering via `MultiSiteSteeringHook`.
- `scripts/aggregate_results_v2.py` — hardened aggregator with per-seed P2 transparency, P5 variance gate, and site-sweep / joint verdicts.

### Modified plan
- `refine-logs/EXPERIMENT_PLAN.md` — M5 section rewritten (v1 → v1+v2 authoritative + v3 logit-supplement + joint distributed-cause test). `resource_fidelity: cost-aware` marker preserved verbatim.

### Regenerated results
- `refine-logs/EXPERIMENT_RESULTS.md` — v2 report (P1 pass, P2 fail, P3 fail, P4 v2 fail, P5 hold; M5-v3 site sweep + joint sections added).
- `results/all_summary_v2.json` — machine-readable roll-up.

### Experiment outputs
- `results/m5_v2/m5v2_diff_of_means_seed{42,123,2024}.json`
- `results/m5_v3/m5v3_expected_score_{E4L10,E1L5,E2L5,E3L10,E3L5}_seed{42,123,2024}.json` (15 files)
- `results/m5_v3_joint/m5v3_joint_expected_score_seed{42,123,2024}.json`

### Run manifests (cost.json each)
- `runs/iteration_round_1/m5v2_full/` (1.665 GPU-h)
- `runs/iteration_round_2/m5v3_logit/` (0.088 GPU-h)
- `runs/iteration_round_3/m5v3_sitesweep/` (0.347 GPU-h)
- `runs/iteration_round_4/m5v3_joint/` (0.187 GPU-h)

### Review-stage artifacts
- `review-stage/AUTO_REVIEW.md` — chronological per-iteration audit log.
- `review-stage/REVIEWER_MEMORY.md` — reviewer's persistent suspicion account (5 iterations).
- `review-stage/REVIEW_STATE.json` — machine state (status=completed, termination_reason=positive_verdict, last_score=7, last_verdict=ready).
- `review-stage/AUTO_ITERATION_FINAL_REPORT.md` — this file.
