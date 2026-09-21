# Auto Iteration Final Report — Belief-Circuit Reproduction on Pythia

- **Generated**: 2026-07-22T10:50:00
- **Iterations consumed**: **0 / 6** (all 5 iterations were ⓪ narrative-only or no-action; type ⓪ does not consume the back-edge budget)
- **Claim-reentries consumed**: **0 / 2** (no ③ actions)
- **Final reviewer score**: **8.8 / 10** (trajectory 8.0 → 8.5 → 8.8 → 8.8 → 8.8)
- **Final canonical verdict**: **almost**
- **Termination reason**: **stalled** (two consecutive no-op iterations with unchanged score/verdict, i4 & i5; corresponds to reviewer's iteration-3 explicit "this is the natural terminal state" declaration)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0 (no new experiments deployed by iteration)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The iteration loop consumed 5 reviewer rounds on a strict Pythia reproduction (`resource_fidelity: strict`, `behavior_source: given`, `mechanism: given`). Two rounds of type-⓪ narrative-only edits addressed all reviewer-flagged reader-interpretation risks (C2 Pythia-family scope, C1 scaling-law overclaim, C3 24/154-checkpoint sparsity, C4 PPL disclosure, "belief heads" over-reification, top-level summary-table scope-flattening). The remaining 3 rounds were no-op polls confirming reviewer stability, culminating in stall-guard termination. No experiments were dispatched, no scripts touched, no claims rewritten. The reviewer's iteration-3 conclusion — *"this is the natural terminal state ... further looping is likely to be churn ... I would not ask for another iteration unless you are allowed to change the verification policy itself or acquire a valid in-family neighbor model"* — sums up the outcome: scientifically stable, mechanically non-green due to C2's out-of-scope cross-family stress test (OLMo-1B) failing under a task.md hard-constraint blocker on in-family Pythia neighbor downloads.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 1 (C2) | 1 SUPPORTED-within-Pythia (⓪ scope-boundary caveat; verify-bookkeeping remains FAIL as out-of-scope stress test) |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 3 (C1, C3, C4) | 3 SUPPORTED-with-caveats + still-INTEGRITY_ONLY (stress-tests deferred by `max_verify_claims_cap`; upgrade commands recorded in Open Items) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

*(No claims entered as PASS.)*

---

## Section 2 — FAIL Claims (full journey)

### 2.1 `C2` — Belief Heads Localization

**Original FAIL signal**
- robustness = 0.00 (0 / 1 eligible variants; threshold 0.50)
- Inconsistent dimensions: cross-family model swap (OLMo-1B) → 0/2 targets localized
  - `personal_belief`: Fisher-ranked heads clear C2a (target drop ≥30%) but violate C2c (off-target drops ≥10%) and C2d (PPL up to 46× clean) — heads entangled with general LM computation
  - `attributed_belief`: all 30 greedy steps show *negative* accuracy drops → same Fisher procedure selects **suppression heads** rather than encoding heads in OLMo-1B (causal direction inverted)
  - Jackknife ρ = 0.954 on OLMo-1B: Fisher signal itself stable → failure is architectural / training-corpus dependent, not statistical
- Variant integrity at entry: 1/1 eligible; variant integrity_status = WARN (cross-vocabulary PPL — informational; C2a/C2c drop-based criteria unaffected)
- Main-experiment integrity: PASS (5/5 admissible Pythia (model, target) pairs LOCALIZED under all four verbatim criteria)

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ⓪ | Verify FAIL will be read as claim contradiction by top-venue reviewers if unfenced | Added explicit Pythia-family boundary caveat + operational-designator terminology note to `refine-logs/FINAL_PROPOSAL.md` §C2, `refine-logs/EXPERIMENT_RESULTS.md` M2, `CLAIMS_LEDGER.md` C2 row | Reviewer i2 confirmed overgeneralization concern RESOLVED; verify-bookkeeping remains FAIL by design |
| 2 | ⓪ | Top-level summary surfaces still implicitly flatten "SUPPORTED-within-Pythia" into "SUPPORTED"; terminology grep needed | Added one-liner directly below `CLAIMS_LEDGER.md` summary table making bookkeeping-vs-scope distinction explicit; C2 Final column changed from "FAIL (fragile under model swap)" to "SUPPORTED-within-Pythia (verify FAIL is out-of-scope cross-family probe)"; all 4 rows in `EXPERIMENT_RESULTS.md` summary table scope-qualified; "belief heads" grep pass audited | Reviewer i3 confirmed both asks satisfied and declared "natural terminal state" |
| 3 | — | none | no-action | Score 8.5 → 8.8 |
| 4 | — | none | no-action | consecutive_noop_count = 1 |
| 5 | — | none | no-action | consecutive_noop_count = 2 → stall guard fires |

**Path taken (summary)**: ⓪ scope-boundary narrative fix (two edit passes) → 3 confirming no-op polls → stall termination. No ① / ② / ③ triggered.

**Experiment & script modifications** (cumulative — all iteration ⓪, no experiment scripts touched)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `refine-logs/FINAL_PROPOSAL.md` §C2 Must-Prove | *"For each model that clears the above-chance gate on the target belief task, the Fisher-mask + zero-ablation search identifies a smallest head set H* satisfying all four verbatim criteria …"* | *"For each **Pythia** model that clears the above-chance gate … **Boundary caveat (iteration ⓪ i1):** the localization result is supported only within the Pythia family. A cross-family swap-test to OLMo-1B did NOT transfer; personal_belief violates C2c/C2d, attributed_belief has inverted causal direction (suppression heads). Jackknife ρ=0.954 confirms failure is architectural, not statistical. Terminology note: 'belief heads' is an operational designator for the Pythia-family Fisher-selected set, not a claim of universal semantic head identity."* |
| 1 | `refine-logs/EXPERIMENT_RESULTS.md` M2 verdict line | `**verdict: SUPPORTED**` | `**verdict: SUPPORTED-within-Pythia**` + new **Cross-family boundary** subsection documenting OLMo-1B 0/2 result |
| 1 | `CLAIMS_LEDGER.md` C2 row | Iteration field "pending"; Final "FAIL (fragile under model swap)"; Caveats only R1 exclusion | Iteration field expanded with the ⓪ explanation + hard-constraint blocker rationale; Final "SUPPORTED-within-Pythia (fragile under cross-family swap …)"; Caveats now include cross-family-fragility summary + terminology note |
| 2 | `CLAIMS_LEDGER.md` summary table C2 Final column | `FAIL (fragile under model swap)` | `SUPPORTED-within-Pythia (verify FAIL is out-of-scope cross-family probe)` |
| 2 | `CLAIMS_LEDGER.md` below summary table | (no bookkeeping note) | Added top-level one-liner: *"Automated verify marks C2 as FAIL because the cross-family model-swap robustness probe (OLMo-1B) failed. This does NOT contradict the paper claim, which is intentionally restricted to the Pythia family — all 5/5 admissible (Pythia model, target) pairs LOCALIZED under all four verbatim criteria. The mechanical FAIL label reflects a stress-test beyond the claim's scope, not a defect in the claim itself."* |
| 2 | `refine-logs/EXPERIMENT_RESULTS.md` summary table all 4 rows | All `**SUPPORTED**` | All scope-qualified (`SUPPORTED (3-scale observation, not a scaling law)` / `SUPPORTED-within-Pythia (cross-family transfer fails on OLMo-1B)` / `SUPPORTED-coarse-window (interval-censored on 24/154 checkpoints)` / `SUPPORTED-on-tested-Pythia-scales (effect size model-dependent)`) + appended verify-state line |

**Claim modifications**
- **Original claim id C2**: "For pythia models that behaviorally clear the above-chance gate, Fisher-information masks (top-0.1% target AND-NOT top-1% F_knowledge) identify a smallest attention-head set H* whose zero-ablation satisfies all four causal-localization criteria …"
- **No rewrite (no ③).** Reviewer at iteration 1 explicitly recommended AGAINST rewriting: *"the current wording is already scoped enough. If you do rewrite, it would just make explicit what is already implicit."* Boundary and terminology clarifications were folded into the same claim text as caveats rather than promoted to a new claim id.

**Final experiment summary**
- **No new runs cited** (iteration budget consumed = 0/6; gpu_hours = 0.0). All updates were narrative caveats to existing docs.
- **Final robustness**: 0.00 on disk (unchanged from verify's original output — verify's Stage 3 was not re-invoked because the ⓪ scope-boundary fix does not create new variant data to re-judge).
- **Final variant pass rate**: 0/1 on disk (unchanged).
- **Final reviewer_status**: **SUPPORTED-within-Pythia** (reviewer's assessment as of iteration 3). Verify's mechanical robustness=0.00 label is retained on disk but is explicitly contextualized as an out-of-scope cross-family stress-test artifact in the top-level surfaces of `CLAIMS_LEDGER.md` and `refine-logs/EXPERIMENT_RESULTS.md`.
- **Final verify_bookkeeping_status**: still **FAIL** (unchanged; hard-constraint blocker prevents any in-family Pythia neighbor variant that could shift this).

**Per-claim final block (structured for ledger merge)**
```
C2:
  final_reviewer_status: SUPPORTED-within-Pythia
  final_verify_bookkeeping_status: FAIL (out-of-scope cross-family probe, definitionally)
  changed:
    - added Pythia-family boundary caveat to FINAL_PROPOSAL.md §C2 Must-Prove (i1)
    - added Cross-family boundary subsection to EXPERIMENT_RESULTS.md M2 (i1)
    - reworded C2 row in CLAIMS_LEDGER.md (Iteration, Final, Caveats fields; i1)
    - added top-level bookkeeping-vs-scope one-liner in CLAIMS_LEDGER.md summary table (i2)
    - scope-qualified all 4 verdict labels in EXPERIMENT_RESULTS.md summary table (i2)
    - added "operational designator, not universal semantic head identity" terminology note (i1)
  falsified: none (the claim's Pythia-scoped wording was NOT falsified; the OLMo-1B result answers a broader question the claim did not make)
  narrowed_to: no formal claim rewrite; the existing Pythia-scoped wording was preserved and the boundary made explicit via caveats
```

**Reviewer memory thread** (C2-specific bullets across iterations)
- i1: C2 scope may be over-generalized by readers; Fisher-signal stability != cross-family transferability; "belief heads" reification risk. → i2: RESOLVED via explicit Pythia scoping + cross-family failure disclosure + operational terminology.
- i2: top-level surfaces still flatten scope; grep pass needed. → i3: RESOLVED — placement is in the right visible location; grep pass sufficient.
- i3–i5: no new C2 suspicions. Reviewer accepts terminal state.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*(No INCONCLUSIVE claims.)*

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*(No ZERO_ELIGIBLE_VARIANTS claims.)*

---

## Section 4b — INTEGRITY_ONLY Claims (no back-edge; recorded as Open Items with upgrade suggestions)

Bucket: `verify_integrity_only = [C1, C3, C4]`, all with `stage2_skip_reason: max_verify_claims_cap` (Stage 1 admitted them, but Phase-3 step-0 picked C2 alone under `MAX_VERIFY_CLAIMS = 1`). Per the bucket contract, iteration takes **no back-edge action** on these three; only paper-side ⓪ caveats + upgrade suggestions.

### 4b.1 `C1` — Scale-Dependent Emergence

- **Original INTEGRITY_ONLY**: `main_experiment_integrity: pass`, Stage 2 skipped
- **⓪ paper-side caveats added (i1)**: "3-scales-only, not a scaling law"; personal non-monotonicity (0.85 → 0.79 → 0.99) precludes any smooth monotonic reading
- **⓪ summary-table scope-qualification (i2)**: Final = `SUPPORTED-across-3-Pythia-scales (awaiting swap-test)`
- **Reviewer consistency check (i1)**: numbers match narrative; recommends avoiding "emergence" language implying smooth scale-law
- **Final reviewer_status**: SUPPORTED-across-3-Pythia-scales
- **Upgrade suggestion (Open Items)**: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run)

### 4b.2 `C3` — Formation Window

- **Original INTEGRITY_ONLY**: `main_experiment_integrity: warn` (`warn_source: experiment` — scope: 24/154 checkpoints), Stage 2 skipped
- **⓪ paper-side caveats added (i1)**: "interval-censored coarse bounds over the observed checkpoint subset, not high-resolution estimates"; personal window [13000, 13000] reflects sampling granularity (adjacent checkpoints step 8000 and step 23000), not a sharp transition; attributed early behavioral window at step 0 reflects operational-definition + prompt-structure bias, not a causal-circuit claim at step 0
- **⓪ summary-table scope-qualification (i2)**: Final = `SUPPORTED-coarse-window (awaiting swap-test)`
- **Reviewer consistency check (i1)**: acceptable if heavily caveated. i2 → RESOLVED.
- **Final reviewer_status**: SUPPORTED-coarse-window
- **Upgrade suggestion (Open Items)**: `/auto-verify C3 -- resume: true` (single-claim mode; the environmental 24/154-checkpoint gap remains — swap-tests over the coarse observed subset are what would be available)

### 4b.3 `C4` — Dynamic Controllability

- **Original INTEGRITY_ONLY**: `main_experiment_integrity: pass`, Stage 2 skipped
- **⓪ paper-side caveats added (i1)**: "WK exactly preserved" is task-level WK accuracy on belief_holdout WK subset only; general LM ability tracked separately via Pile PPL (1.047× on pythia-1b, 1.005× on pythia-2.8b — both under C2 1.05× bar, non-zero); model-dependent effect size (+151 vs +27) precludes claim of uniform controllability across scale
- **⓪ summary-table scope-qualification (i2)**: Final = `SUPPORTED-on-tested-Pythia-scales (awaiting swap-test)`
- **Reviewer consistency check (i1)**: numbers match narrative; PPL disclosure was the missing piece. i2 → RESOLVED.
- **Final reviewer_status**: SUPPORTED-on-tested-Pythia-scales
- **Upgrade suggestion (Open Items)**: `/auto-verify C4 -- resume: true`

**Per-claim final block (structured for ledger merge)**
```
C1:
  final_reviewer_status: SUPPORTED-across-3-Pythia-scales
  final_verify_bookkeeping_status: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
  changed: [added "3-scales-only, not a scaling law" caveat; scope-qualified Final label]
  falsified: none
  narrowed_to: no formal rewrite; scope caveats added to existing claim
C3:
  final_reviewer_status: SUPPORTED-coarse-window
  final_verify_bookkeeping_status: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap; main-experiment warn from 24/154 scope)
  changed: [added interval-censored coarse-bounds caveat; clarified singleton [13000,13000] and step-0 attributed]
  falsified: none
  narrowed_to: no formal rewrite; coarse-window scoping added as caveats
C4:
  final_reviewer_status: SUPPORTED-on-tested-Pythia-scales
  final_verify_bookkeeping_status: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
  changed: [added PPL disclosure; scale-nonuniform-effect-size caveat]
  falsified: none
  narrowed_to: no formal rewrite; scale-scope caveat added
```

---

## Section 5 — Legacy DEFERRED Claims

*(Empty — new architecture; no legacy deferred_claims bucket populated.)*

---

## Section 6 — Cross-Cutting Patterns

- **Fisher signal stability ≠ functional transferability** (flagged i1; recurs across the C2 analysis). Jackknife ρ ∈ [0.885, 0.992] on Pythia AND ρ = 0.954 on OLMo-1B both show stable Fisher rankings — but the *causal function* of the top-Fisher heads differs by model family (encoding in Pythia, suppressing in OLMo-1B for attributed_belief). This is a systemic risk for any Fisher-based mechanism-attribution study: **must not conflate ranking stability with functional generalization**. Status at termination: **surfaced but architectural — cannot be resolved without more model families**.
- **Reproduction-fidelity strictness restricts variant options** (flagged i1). When the claim is family-scoped and the environment lacks in-family neighbors, verify's swap-test can produce a definitionally out-of-scope FAIL. Clean remediation is boundary-clarification (⓪), not method redesign. Status at termination: **applied — C2 fenced with Pythia-family boundary caveat**.
- **Reader-interpretation risk from unqualified verdict labels** (flagged i2). Even scientifically-correct claims can be misread when top-level surfaces flatten scope-qualified verdicts into unqualified "SUPPORTED". Status at termination: **resolved — all summary-table surfaces now carry scope-qualified verdicts + explicit bookkeeping-vs-scope one-liner**.
- **Presentation-vs-mechanical-bookkeeping mismatch under hard constraints** (surfaced i3). Under reproduction-strictness + environmental blockers, the mechanical STOP rule (verify_failed empty) can be structurally unreachable while the science-level assessment is fully settled. The correct outcome is termination on the stall guard, not further looping or fake experimental escalation. Status at termination: **applied — loop terminated cleanly via stall guard**.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: **0 / 6** (all 5 iterations were ⓪ narrative-only or no-action; ⓪ does not consume the back-edge budget)
- **Claim-reentries consumed**: **0 / 2** (no ③ actions)
- **Iteration `/run-experiment` calls**: 0
- **Iteration GPU-hours**: 0.0
- **Consecutive no-op count at termination**: 2 (i4 and i5)

### Per-iteration breakdown (all ⓪ or no-action; `iteration_breakdown[]` in state file is empty by design — the ⓪ + no-action entries do not populate that array per the skill's Phase C contract)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hrs | Score after | Verdict after | consecutive_noop_count after |
|---|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative | C1, C2, C3, C4 | — | 0 | 0.0 | 8.0 | almost | 0 |
| 2 | ⓪ narrative | C1, C2, C3, C4 | — | 0 | 0.0 | 8.5 | almost | 0 (score changed) |
| 3 | — no-action | — | — | 0 | 0.0 | 8.8 | almost | 0 (score changed) |
| 4 | — no-action | — | — | 0 | 0.0 | 8.8 | almost | 1 (⓪-only + unchanged) |
| 5 | — no-action | — | — | 0 | 0.0 | 8.8 | almost | 2 → **stall trigger** |

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims** (after exhausting routing options):
  - `C2` — verify-bookkeeping remains FAIL as a definitional out-of-scope cross-family stress-test artifact (OLMo-1B model swap failed to localize; the same Fisher procedure selects suppression heads in OLMo-1B for attributed_belief, inverted causal direction). Task.md hard constraint restricts on-disk Pythia weights to {410m, 1b, 2.8b}; downloading pythia-160m or pythia-6.9b is blocked. **No admissible in-family same-family model-swap variant available under current environment.** Reviewer's iteration-1 recommendation: ⓪ narrative-only (applied). Scientific interpretation at termination: **SUPPORTED-within-Pythia**.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)** — carry forward with per-`stage2_skip_reason` upgrade commands:
  - `C1` [`stage2_skip_reason: max_verify_claims_cap`]: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). `main_experiment_integrity: pass`.
  - `C3` [`stage2_skip_reason: max_verify_claims_cap`]: `/auto-verify C3 -- resume: true`. `main_experiment_integrity: warn`; `warn_source: experiment` (scope: 24/154 checkpoints — coarse-window resolution).
  - `C4` [`stage2_skip_reason: max_verify_claims_cap`]: `/auto-verify C4 -- resume: true`. `main_experiment_integrity: pass`.
- **Legacy deferred claims**: none.
- **Recurring unresolved patterns**:
  - Fisher signal stability ≠ functional transferability (surfaced but architectural — cannot be resolved without additional Pythia scales or cross-family comparisons that survive the same integrity criteria).
- **Claim-reentry refusals**: none (reviewer never requested ③).
- **Environmental / process-level items surfaced during iteration (not iteration-fixable)**:
  - C3 environmental gap: 24 of 154 planned pythia-1b intermediate checkpoints on disk (native log-spaced subset). Not a downscale (missing files, not cost-saving), so strict-harness HALT rule inapplicable. Coarse-log-spaced resolution is inherent to the available data.
  - Cross-model code review substitution during experiment stage: llm-chat MCP was unavailable during Phase 3 → self-review + hook sanity tests (`scripts/_test_hooks.py`, `scripts/_test_logprob.py`) substituted per CLAIMS_LEDGER.md precedent. One bug caught during self-review: batched `continuation_logprob` shared KV-cache across continuations under `transformers 4.57` stateful `Cache` — fixed by dropping the KV-cache optimization.
  - Pipeline-tooling gap: the mechanical STOP rule requires `verify_failed` to be empty; under reproduction-strictness + environmental hard constraints there is no admissible action that can clear a definitionally out-of-scope cross-family FAIL. This is a policy/tooling consideration, not a scientific defect — as reviewer i3 explicitly stated: *"If some external system requires verify_failed == empty, then the only honest resolution is a policy/tooling change, not more paper edits and not fake experimental escalation."*
