# Auto Iteration Final Report — Faithful SAGE Reproduction

- **Generated**: 2026-07-14T03:36:00Z
- **Iterations consumed**: 0 / 6 (only type-⓪ narrative-only actions in iteration 1; no back-edge action fired)
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: almost
- **Termination reason**: `positive_verdict` (three-dimensional STOP rule fired at iteration 1)
- **Cumulative iteration cost**: runs_total = 0, gpu_hours_total = 0.00
- **Cumulative pipeline cost** (main + verify + iteration): 1.68 GPU-hours of 10-hour budget; 8.32 GPU-hours remain unspent
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The iteration loop terminated at iteration 1 on a positive verdict (score 6/10, "almost"). The reviewer's assessment is that the reproduction is **scientifically honest and publishable as a mostly-null / attribution-confounded reproduction**, but must not be sold as a positive replication of SAGE's claimed advantages. All four claims (C1-C4) are either PASS (C4 — the null verdict on cross-pair generalization is robust across two architecturally distinct pairs) or INTEGRITY_ONLY (C1, C2, C3 — Stage 1 audits admitted them, Stage 2 swap-tests were deferred by `MAX_VERIFY_CLAIMS=1` cap). No FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claims exist, so the third dimension of the STOP rule was satisfied on entry. The reviewer identified narrative/framing corrections (type ⓪) as the paper-side actions needed before submission, plus optional INTEGRITY_ONLY upgrades (priority order C1 > C2 > C3; each ~1 GPU-hr from remaining 8.32) that the human paper author can dispatch if desired — these are outside the iteration loop's scope by design.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C4) | 1 PASS (held) |
| FAIL                     | 0      | — |
| INCONCLUSIVE             | 0      | — |
| ZERO_ELIGIBLE_VARIANTS   | 0      | — |
| INTEGRITY_ONLY (max_verify_claims_cap) | 3 (C1, C2, C3) | 3 still INTEGRITY_ONLY (no back-edge; upgrade deferred to human) |
| DEFERRED (legacy)        | 0      | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C4` — Cross-LLM+SAE-pair generalization (Qwen3-4B main; GPT-OSS-20B variant)

- **Original robustness signal**: robustness = 1.00, variants_passed = 1/1 (model-swap-gpt-oss-20b: Δ Pearson = -0.045, n.s. — consistent with main experiment's Δ = +0.153, n.s.).
- **Reviewer consistency check**: narrative is honest **if phrased carefully**; numeric consistency confirmed. The correct paper wording is: *"Across two cross-LLM+SAE pairs, SAGE-lite does not significantly outperform the Neuronpedia proxy on predictive accuracy; this null result appears robust to swapping the cross-pair variant."*
- **Touched in iterations**: [1] — narrative-only reframing recorded.
- **Final status**: PASS (held) — but note that "PASS" here is *internal ledger language*; the honest scientific verdict is a robust null, not a cross-pair success.
- **Notes for downstream (must appear in the paper)**:
  1. Do **not** write "robust across pairs" in a way that implies consistent effect direction — the two Δ Pearson values have opposite sign (+0.153 vs -0.045). What is robust is the *null verdict*, not the *effect*.
  2. This is **SAGE-lite predictive-only** (Explainer + Reviewer only, no Designer + Analyzer; no generative-accuracy test on Qwen3-4B). Every C4 mention must state this scope qualifier.
  3. Paper language should read "no statistically significant cross-pair advantage on either evaluated pair," never "PASS on cross-pair" or "SAGE generalizes across model/SAE pairs".
  4. Do not imply Stage 2 verification proves broad cross-pair failure — it supports failure on these two tested pairs, under this SAGE-lite setup and this sample size (n=34 / n=45).

---

## Section 2 — FAIL Claims (full journey)

*(none in this run — no claim was FAIL after `/auto-verify`.)*

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*(none in this run — no claim was INCONCLUSIVE after `/auto-verify`.)*

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*(none in this run — Phase 9 variant integrity passed for all variants that ran.)*

### 4b — INTEGRITY_ONLY Claims (Stage 2 deferred by `max_verify_claims_cap`)

The iteration loop does NOT act on INTEGRITY_ONLY claims by design — they are no-action-with-upgrade-suggestion outputs (Stage 1 audit passed with warnings; Stage 2 swap-tests intentionally skipped by the `MAX_VERIFY_CLAIMS=1` cap). The reviewer's recommendation is that the human paper author dispatches upgrades in the priority order C1 > C2 > C3, budget-permitting, before submission.

#### 4b.1 `C1` — SAGE beats Neuronpedia on generative accuracy (main pair)

- **Main-experiment integrity**: WARN — `warn_source: experiment (statistical resolution: Wilcoxon n_eff=4 / 90.9% zero-differences in gen_acc; 5 probes/feature is a coarse discrete metric).`
- **Main-experiment verdict**: partial-support [attribution-confounded] — Δgen_acc (SAGE − Neuronpedia) = +0.018, CI [+0.005, +0.036], Wilcoxon p=0.045 **SIGNIFICANT vs Neuronpedia**; BUT Δgen_acc (SAGE − matched-backbone GPT-5-1shot) = -0.005, CI [-0.032, +0.023], p=0.74 **NOT SIGNIFICANT**. Under `underpower: tag` — realized n = 44 / 300 planned.
- **`stage2_skip_reason`**: `max_verify_claims_cap` — C4 was picked as the top-1 admitted claim.
- **Upgrade command**: `/auto-verify C1 — resume: true, swap-variants: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2-3 run). Estimated cost: ~1 GPU-hr.
- **Reviewer priority**: **highest** — this is the only "positive" headline, is known to be attribution-confounded, and Stage 2 swap-tests would close the argument.
- **Paper-side framing action (already recorded in iteration 1, type ⓪)**: split C1 into
  - **C1a**: SAGE exceeds the public Neuronpedia catalog built with an older GPT-4o-mini backbone.
  - **C1b**: SAGE does **not** significantly exceed a matched-backbone GPT-5 single-pass baseline.
  - Promote **C1b** in abstract/conclusion.

#### 4b.2 `C2` — SAGE beats Neuronpedia on predictive accuracy (main pair)

- **Main-experiment integrity**: PASS.
- **Main-experiment verdict**: not-supported — Δ pearson (SAGE − Neuronpedia) = -0.018, CI [-0.093, +0.058], p=0.43. Under `underpower: tag` — realized n = 44 / 300 planned.
- **`stage2_skip_reason`**: `max_verify_claims_cap`.
- **Upgrade command**: `/auto-verify C2 — resume: true, swap-variants: true`.
- **Reviewer priority**: second — a robust null on predictive accuracy would strengthen the reproduction narrative.
- **Paper-side framing action (already recorded in iteration 1, type ⓪)**: describe C2 as a failed replication / null result; make it prominent, not buried behind C1 language.

#### 4b.3 `C3` — Layer-depth generalization

- **Main-experiment integrity**: WARN — `warn_source: experiment (Bonferroni correction described in EXPERIMENT_RESULTS.md but not explicitly implemented in aggregate script; per-layer n=14-15; L20 gen_acc all-zero differences).`
- **Main-experiment verdict**: not-supported — no per-layer CI excludes zero at L4/L12/L20 for either metric at n=14-15/layer.
- **`stage2_skip_reason`**: `max_verify_claims_cap`.
- **Upgrade command**: `/auto-verify C3 — resume: true, swap-variants: true` (best combined with a larger n_features_per_layer rerun of M1 and L20 l0_71 SAE swap).
- **Reviewer priority**: **lowest** — the underlying issue is under-power at n=14-15/depth, not integrity; verification alone will not fix the depth analysis.
- **Paper-side framing action (already recorded in iteration 1, type ⓪)**: describe C3 as unresolved due to under-power, not as evidence for absence of depth dependence. If ever upgraded, swap in the L20 `l0_71` SAE variant.

---

## Section 5 — Legacy DEFERRED Claims

*(empty under current architecture; retained header for backward compatibility.)*

---

## Section 6 — Cross-Cutting Patterns

- **Ledger vocabulary vs. paper vocabulary gap** (highest-risk overclaim source): The pipeline's internal PASS/FAIL/INTEGRITY_ONLY vocabulary diverges from the honest scientific narrative. "PASS" on C4 means "null verdict robust", **not** "SAGE succeeded on cross-pair". "INTEGRITY_ONLY" on C1/C2/C3 means "Stage 1 admits, Stage 2 skipped by cap", **not** "these claims are verified". The paper must translate ledger language into scientific language every time. Reviewer will monitor this in future iterations.
- **Systemic under-power** (touches all four claims — C1, C2, C3, C4): every claim carries `suspected_under_power: true` because DMXAPI throughput variance capped realized n at 15-23 % of planned. This is not a per-claim methodology issue; it is a session-level infrastructure constraint. It must be treated as a major inferential limitation in the paper, not an operational nuisance. Not resolvable within the iteration loop's remaining budget without a higher-QPS GPT-5 backbone.
- **Attribution-confound on the headline positive** (touches C1, and reframes C4's positive-trend on Qwen3-4B): the M0.5 gate discovered that single-pass GPT-5 already outscores Neuronpedia's public GPT-4o-mini explanations. Any "SAGE > Neuronpedia" headline must qualify with "against the specific public older-backbone catalog, not against a matched-backbone GPT-5 baseline". The matched-backbone comparator is the honest scientific control and must be elevated to primary in the paper. **Recorded but not fully resolved** — depends on paper-side framing execution.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6 (iteration 1 fired only type-⓪ narrative-only actions; no back-edge action ran)
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.00
- **Cumulative pipeline cost**: main experiment 1.66 GPU-hours + verify 0.02 GPU-hours + iteration 0.00 GPU-hours = **1.68 GPU-hours** of 10-hour budget; **8.32 GPU-hours remain unspent**.

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only | C1, C2, C3, C4 (all paper-side wording corrections) | — | 0 | 0.00 | 6/10 | almost |

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close. These need a human or a separate pipeline.

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)** — from iteration 1's `Open Items — Unverified Under Swaps`:
  1. **C1** — `stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment (Wilcoxon n_eff=4, coarse 5-probe gen_acc metric)`. Upgrade: `/auto-verify C1 — resume: true, swap-variants: true`. **Priority: highest** (~1 GPU-hr).
  2. **C2** — `stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: pass`. Upgrade: `/auto-verify C2 — resume: true, swap-variants: true`. **Priority: second** (~1 GPU-hr).
  3. **C3** — `stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment (Bonferroni framing not coded; per-layer n=14-15; L20 gen_acc all-zero deltas)`. Upgrade: `/auto-verify C3 — resume: true, swap-variants: true` (best combined with a larger n_features_per_layer rerun of M1 and L20 l0_71 SAE swap). **Priority: lowest** — underlying issue is under-power, not integrity.
- **Legacy deferred claims (empty in new runs)**: none.
- **Recurring unresolved patterns**:
  - Ledger-vs-paper vocabulary gap (must be translated in every claim mention).
  - Systemic under-power (session-level infrastructure constraint; needs higher-QPS backbone to resolve, out of scope for this loop).
  - Attribution-confound on the headline positive (paper-side framing correction recorded as type-⓪; execution is the paper author's responsibility).
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none — the reviewer did not request any type-③ claim rewrite. (The C1a/C1b split is a *presentation* structure, not a new claim id; both comparators are already in `results/aggregate.json`.)

### Paper-side wording checklist (from reviewer, all type ⓪)

- [ ] Recast C1 as **C1a** (vs public Neuronpedia older backbone) + **C1b** (vs matched-backbone GPT-5-1shot); promote **C1b** in abstract/conclusion.
- [ ] Describe C2 as a **failed replication / null result**; make it prominent.
- [ ] Describe C4 as a **verified negative transfer result across two pairs**; state that effect directions differ (+0.153 vs -0.045) and only the null verdict is robust; state SAGE-lite predictive-only scope every time.
- [ ] Describe C3 as **unresolved due to under-power**, not as evidence for absence of depth dependence.
- [ ] Do **not** write "PASS" on C4 as if it meant SAGE succeeded cross-pair.
- [ ] Do **not** oversell "methodology gate passed" — it is an internal gate, not scientific replication.
- [ ] Elevate the matched-backbone GPT-5-1shot control to primary comparator; Neuronpedia public catalog is a historical baseline.
- [ ] State under-power as a major inferential limitation, not an operational nuisance.
