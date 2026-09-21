# Auto Iteration Final Report — Steering Evo2-7B toward high α-helical content

- **Generated**: 2026-08-24T09:10:30Z
- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 7 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict
- **Cumulative cost**: runs_total=0, gpu_hours_total=0.0 (no back-edge experiments; iteration-stage budget untouched — ~11.2 GPU-h of the whole-round 40-h cap remain)
- **Reviewer**: external `gpt-5.4` via the skill's documented HTTP fallback (the MCP `llm-chat` tool was not exposed in this host context, but the endpoint was reachable and credentials were environment-provided — a genuine external review, not the self-review degradation the NOTICE flagged as possible)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The verify stage handed the iteration loop a clean, converged result: C1 (helix gain / dose-response)
PASS at robustness 1.00, C2 (validity non-inferior) INTEGRITY_ONLY (swap-test deferred under the
MAX_VERIFY_CLAIMS=1 cap, with a Phase-2 wording WARN). No claim was in FAIL / INCONCLUSIVE /
ZERO_ELIGIBLE_VARIANTS, so no back-edge (①/②/③) was warranted or fired. The external reviewer scored the
work 7/10, "almost" ready, confirmed C1's numbers are internally consistent, and judged that the single
actionable issue — C2's "off-target properties not degraded" wording overclaiming against the documented
−37% protein-length and −0.139 GC shift — is a paper-side (type ⓪) tightening, which was applied to the
authoritative claim statement. The three-dimensional STOP rule fired on iteration 1 (score 7 ≥ 6, verdict
`almost`, all blocking buckets empty). No claims remain in a blocking state; C2's deferred swap-test is
carried forward as an optional Open Item.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C1) | PASS (held) |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 1 (C2) | INTEGRITY_ONLY held; overclaim wording superseded (⓪); swap-test deferred → Open Items |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C1` — CAA steering at residual site 28 raises α-helix fraction with a monotone dose-response
- **Original robustness signal**: robustness=1.00, variants_passed=1/1 (model-swap `evo2_7b`→`evo2_7b_262k` reproduced ρ=0.878, winning-coef gain +0.094, δ=0.201).
- **Reviewer consistency check**: PASS. Dose ladder (0.414/0.475/0.482/0.524/0.575/0.563) monotone through coef 4, consistent with ρ=0.922 / q=1.5e-4; winning +0.068 matches bootstrap CI [+0.040,+0.098]; specificity (matched −0.024 n.s., sham −0.005 n.s.; CAA−control +0.093 p=9e-10) and length-matched +0.089 all reconcile with `refine-logs/EXPERIMENT_RESULTS.md` and `verify/C1_helix_gain_doseresponse/ROBUSTNESS.md`. No numeric drift.
- **Touched in iterations**: [1] (consistency check only)
- **Final status**: PASS (held)
- **Notes for downstream**: paper-side caveats to carry — (a) modest per-generation effect size (Cliff's δ=0.135, Hedges g=0.237); (b) length/GC off-target coupling is a property of the CAA lever and transfers across checkpoints — frame honestly, not dismissively; (c) proxy-heavy endpoint (DNA→ORF heuristic→ESMFold→DSSP→pLDDT) — frame as computational structural steering, not in-vitro helicity; (d) site specificity (site-28 works, site-30 does not, SAE-clamp negative) is interesting but mechanistically incomplete.

---

## Section 2 — FAIL Claims (full journey)

none — no claim entered in FAIL state.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

none — no claim entered in INCONCLUSIVE state.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

none — no claim entered in ZERO_ELIGIBLE_VARIANTS state.

### Section 4b — INTEGRITY_ONLY Claims (no-action-with-upgrade-suggestion)

### 4b.1 `C2` — validity non-inferior at the winning setting; off-target shifts documented
- **Original state**: INTEGRITY_ONLY, `stage2_skip_reason: max_verify_claims_cap` (admitted at Stage 1 but not the top-1-by-importance pick under MAX_VERIFY_CLAIMS=1). Main-experiment integrity=WARN, warn_source=experiment.
- **Reviewer judgment**: the pre-registered non-inferiority claim itself is methodologically clean (validity 0.968 vs 0.991, one-sided 95% LB −0.035 > margin −0.05). The WARN is entirely about the "off-target properties not degraded" wording overclaiming against the documented −37% length / −0.139 GC shift.
- **Action taken (type ⓪, no budget consumed)**: tightened the authoritative C2 claim statement in `CLAIMS_LEDGER.md` to state that validity is preserved within the NI margin but other sequence properties are not — GC (−0.139) and inferred protein length (−37%) shift substantially as documented CAA-specific off-target effects. This resolves the substance of the Phase-2 WARN on the paper surface. No script, data, or experiment changes; no `/auto-verify` invoked (INTEGRITY_ONLY is a no-back-edge bucket by contract).
- **Final status**: INTEGRITY_ONLY held (audit passed with WARN; wording superseded). Swap-test remains deferred → Open Items.
- **Upgrade suggestion (optional, non-blocking)**: `/auto-verify C2 — resume: true` (single-claim mode; Phase-2 audit reused via RESUME; only Stages 2–3 run) — would stress-test the length-regressed endpoint / alternate pLDDT policy. Fits inside the ~11.2 GPU-h remaining if budget/venue warrant it.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

none.

---

## Section 6 — Cross-Cutting Patterns

- **Off-target coupling is intrinsic to the CAA lever.** The length (−37%) and GC (−0.139) shifts recur across the experiment, verify, and review stages, and the verify model-swap showed the signature transfers to `evo2_7b_262k` → a robust family-level property, not a one-checkpoint artifact. Touched claims: C1 (survives length-matching, no composition bias — so the helix gain is not explained by it) and C2 (the overclaim wording was this pattern's most concrete symptom). Status at termination: **resolved on the paper surface** (C2 wording tightened; C1 already carries the honest framing), but it remains an intrinsic property of the method that the paper must present plainly.
- **Proxy-heavy structural endpoint** (DNA→ORF→ESMFold→DSSP→pLDDT) and **within-family-only generalization** are open framing/scope limits, not defects — carried forward for the write-up.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0

### Per-iteration breakdown (from `iteration_breakdown[]`)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only (C2 wording) | C1 (audit), C2 (⓪) | — | 0 | 0.0 | 7 | almost |

(Iteration 1 consumed no budget: the only action was type ⓪, which is free by contract; the STOP rule
fired the same iteration because all blocking buckets were already empty at entry.)

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - `C2` — `stage2_skip_reason: max_verify_claims_cap`; upgrade: `/auto-verify C2 — resume: true` (single-claim mode; Phase-2 audit reused via RESUME; only Stages 2–3 run). main_experiment_integrity: warn; warn_source: experiment (off-target "not degraded" wording — substance addressed this iteration via the ⓪ tightening; the swap-test on the length-regressed endpoint / alternate pLDDT policy is the remaining optional robustness check).
- **Legacy deferred claims**: none
- **Recurring unresolved patterns**: proxy-heavy structural endpoint limits how strongly biological helicity can be claimed; generalization demonstrated only within the Evo2-7B family (no cross-family model or wet-lab validation). Both are write-up scope caveats, not blockers.
- **Claim-reentry refusals** (③ requested but sub-budget exhausted): none — no ③ was requested (no FAIL claim existed).
