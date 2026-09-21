# Auto Iteration Final Report — Feature Steering an α-Helix Knob in Evo2-7B

- **Generated**: 2026-07-18T15:45:00
- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6.8 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict (3-D STOP: score ≥ 6, verdict ∈ {ready, almost}, no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS remaining)
- **Cumulative cost**: runs_total = 3 (33 random-direction cells across 3 GPU processes), gpu_hours_total ≈ 11.7 (6.14 productive fixed batch + 5.35 wasted mkdssp-PATH-bug run + ~0.2 pilots/debug). gpu_ids witnessed = {3,4,5} ⊆ {2,3,4,5} (GPU 2 held by weiyunx, never used).
- **Reviewer**: external LLM gpt-5.4 via llm-chat MCP.
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The iteration loop closed the single blocking claim (C3, INCONCLUSIVE) with one type-② main-experiment mechanism-harness fix and rode two free type-⓪ narrative relabels alongside it. C3's specificity evidence was rebuilt on clean footing: the decisive statistic was re-locked from the capability-crashed dose α=16 to the capability-preserved mid-plateau α=8, and a genuine 33-direction magnitude-matched random-direction null was added as the primary specificity test — S's α-helix rise (+0.096) exceeds all 33 random directions (empirical p=0.029, z=3.02) with generation quality preserved. The β-sheet-amplification arm was re-stated honestly as a negative result rather than half a double dissociation. A re-run mechanism-audit moved C3 FAIL→WARN, so C3 went INCONCLUSIVE→INTEGRITY_ONLY. C1 was relabeled established→conditional (set-level) with the degenerate seed_jaccard=0.0 disclosed and the eukaryote leg corrected to complete-and-positive; C2's PASS held with its proxy-readout caveats carried. The reviewer moved 5.5→6.8 ("almost"), and the loop terminated on a positive verdict with no claim left in a blocking state.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C2) | 1 PASS (held) |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 1 (C3) | 1 → INTEGRITY_ONLY (main-experiment ② fix: mechanism-audit FAIL→WARN) |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY (entry)   | 1 (C1) | 1 INTEGRITY_ONLY (⓪ relabel; eukaryote correction) |
| DEFERRED                 | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C2` — Amplifying S causally raises encoded-protein α-helix fraction (dose-response)
- **Original robustness signal**: robustness=1.00, variants_passed=1/1 (method-swap mkdssp→pydssp, ρ=0.886 vs main ρ=0.759; effect +0.327 vs +0.332).
- **Reviewer consistency check**: numbers consistent with `m2_dose_response_curve.json`; reviewer called C2 "the strongest part of the paper."
- **Touched in iterations**: [1] (⓪ caveats only)
- **Final status**: PASS (held)
- **Notes for downstream (paper-side caveats, mandatory)**: (a) PROXY readout — ESMFold+DSSP on generated sequences, not experimental structure; frame as a causal effect on a predicted structural phenotype under one predictor stack; (b) structure-PREDICTOR swap axis (ESMFold→OmegaFold) untested (network-infeasible) — robust only to the SS-assignment axis; (c) α*=32 is a grid edge (still rising), not a mapped interior plateau; (d) valid-ORF dips to 0.65 at α=16 then recovers 0.78 at α=32 — non-monotone, discuss explicitly; clean operating point α=8 (+0.10 at baseline quality).

---

## Section 2 — FAIL Claims (full journey)

None. (No claim entered verify in the FAIL state.)

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

### 3.1 `C3` — Amplification effect is specific to S / causally manipulable knob

**Original INCONCLUSIVE reason** (from `verify/C3_specificity_double_dissociation/ROBUSTNESS.md` + `main_experiment_audit/MECHANISM_AUDIT.pre_iteration_fix.md`):
- Main-experiment mechanism-audit **FAIL**: the decisive double-dissociation statistics (S-vs-matched helix p=3.2e-23) were computed at α=16, the MAX of M3's own grid — a dose where the S arm's valid-ORF had crashed 0.888→0.667 (~25% rel, >2× the ~10% tolerance). No plateau located; no genuine random-direction control (only one fixed matched-magnitude direction). β-arm never rose yet a symmetric "double dissociation" was claimed.

**Experiment plan & script modifications**

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `refine-logs/EXPERIMENT_PLAN.md` M3 Controls/Pass-criteria | matched-control as the specificity comparator; decisive dose = grid behavior; "double dissociation" | random-direction null (≥30, norm-matched) as PRIMARY control; decisive dose LOCKED at capability-preserved α=8; β-arm framed as a documented negative; PASS criterion = S is a significant outlier vs the random null at a capability-preserved dose |
| 1 | `code/m3_random_control.py` (new) | — | 33 independent random directions: 19 SAE features drawn uniformly excluding S∪β∪matched, weighted by natural activation magnitude, rescaled to match ‖S.base_dir‖; identical DNA→ESMFold→DSSP pipeline; two-phase (all Evo2 gen → all ESMFold) to avoid runtime alternation |
| 1 | `code/m3_consolidate_v2.py` (new) | `top = max(alphas)` (decisive dose = grid max) → `m3_specificity_summary.json` | locks decisive dose at α=8; S-vs-random-null empirical p + z as primary; matched-control secondary; β-arm negative → `results/m3_specificity_summary_v2.json` |
| 1 | `code/launch_random_control.sh` (new) | — | detached launcher; **bug fix**: `export PATH=<scientist>/bin:$PATH` so the fold readout's `mkdssp` subprocess resolves (root cause of an initial all-null DSSP readout) |

**Re-experiment outcome**

| Iter | Path | New runs | Result |
|---|---|---|---|
| 1 | new random-control batch (`/mechanism-audit` re-run for the verdict) | `runs/iteration_round_1/rand_d{0-10,11-21,22-32}/` (33 directions, GPUs 3/4/5, ~6.14 GPU-h) | mechanism-audit **FAIL→WARN**; combined integrity WARN → C3 **INCONCLUSIVE→INTEGRITY_ONLY** |

**Fixed decisive result (α=8, `results/m3_specificity_summary_v2.json`)**
- S helix Δ = **+0.096** (0.589 vs baseline 0.493); matched −0.003, β −0.010 (both flat).
- PRIMARY random-direction null (N=33): random Δ mean −0.014, sd 0.037, **max +0.061**; **0/33 ≥ S** → empirical one-sided **p=0.029**, **z=3.02**.
- Capability-selection artifact ruled out: S valid-ORF 0.888 ≈ random-null mean 0.887.
- Secondary matched-control: S−matched Δ=+0.099, Mann-Whitney p=3.5e-4.
- β-arm: **NEGATIVE** (Δsheet +0.005) — reported honestly; C3 is helix-axis specificity + β-does-not-raise-helix, not a symmetric double dissociation.

**Claim modifications** (mandatory subsection): none — INCONCLUSIVE routing fixes the main experiment, not the claim wording. The claim statement is unchanged; only its evidential basis was rebuilt and its framing narrowed (double dissociation → helix-axis specificity) in the supporting records.

**Final status**: **INTEGRITY_ONLY** (repaired from INCONCLUSIVE). Specificity SUPPORTED at a capability-preserved dose vs a genuine random null; swap-robustness for C3 not evaluated (MAX_VERIFY_CLAIMS cap) — flagged in Open Items.

**A real bug was caught and fixed, not papered over.** The first random-control batch returned `n_folded_gated=0` for all 33 directions (helix_mean null) despite valid-ORF ≈ 0.88. Diagnosis: ESMFold folded fine (a reproducer showed plddt=75.2), but `dssp_fractions`' `subprocess.run(["mkdssp", ...])` raised `FileNotFoundError` because the detached launcher inherited base-conda's PATH, not the scientist env where `mkdssp` lives (the original M3 runs worked only because `dispatch.py` inherited an activated shell). Fixed by putting the env bin on PATH; verified with a 1-direction pilot (n_folded=10/12, helix 0.493 ≈ baseline — the expected null) and a live-worker `/proc/<pid>/environ` check; the full batch then re-ran clean (all 33 directions folded, min 38 gated structures each).

**Reviewer memory thread (C3)**
- iter1: "decisive stat at α=16 is invalid (capability-degraded); need a genuine ≥30 random-direction null as the primary statistic; β-arm is a negative result." → iter2: all three genuinely addressed ("the comparison I wanted"); residual: N=33 null is acceptable-not-overwhelming (p=0.029 is the zero-exceedance extreme), α=8 at the capability-preserved edge, single predictor stack.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

None.

---

## Section 5 — Legacy DEFERRED Claims

None (empty under the current architecture).

---

## Section 4b / Integrity-only — C1 (entered INTEGRITY_ONLY; ⓪ relabel journey)

### `C1` — α-helix-selective Layer-26 SAE feature set exists (M0)
- **Entry state**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap; main-experiment integrity WARN).
- **Iteration action**: type ⓪ (narrative honesty relabel — no scripts, no runs).
  - Softened `main_experiment.verdict` "established (set-level)" → "conditional (set-level)" per the plan's own four-state vocabulary. Basis made explicit: conditional on set-vs-single-feature + label-honesty grounds (no single feature clears τ=0.75), **NOT** organism restriction.
  - Disclosed the degenerate pre-registered seed_jaccard=0.0 (computed on the always-empty strict single-feature set); the substantive stability is set-level AUROC (0.88-0.90 across seeds/helix-defs) + the actual relaxed-set cross-seed Jaccard ~0.7-0.8.
  - **Eukaryote correction** (team-lead flag): the cross-organism leg is COMPLETE & POSITIVE, not pending — all 6 H. sapiens configs established (350,268 codons, set AUROC 0.858, confound-only 0.517, null 0.506, 341 FDR-sig). Cross-organism generality confirmed (E. coli + H. sapiens).
- **Final status**: INTEGRITY_ONLY (label now honest; phenomenon real and general). Swap-robustness deferred by the verify cap — Open Items.

---

## Section 6 — Cross-Cutting Patterns

- **Rhetoric outran the cleanest evidence, then was reeled back after audit** (touched C1, C2, C3). Recurrent across the project: decisive stats taken at edge-of-grid / capability-degraded doses (C3 α=16, C2 α*=32); a favorable statistic substituted for a degenerate pre-registered one (C1 seed_jaccard); "double dissociation" claimed when only one arm moved (C3 β). **Resolution status at termination**: substantively addressed for the blocking claim — in iteration 1 the authors corrected the decisive *statistic* (re-lock + genuine null), not merely the phrasing; the reviewer explicitly credited this as improving confidence. The pattern is not fully eliminated (it cost trust and required post-hoc corrections) but is no longer producing a blocking overclaim.
- **Operational pipeline fragility** (C3): the mkdssp-PATH bug silently zeroed a whole readout; disclosed and fixed honestly. Non-blocking, but flagged as a stack-robustness consideration.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 3 (33 random-direction cells across 3 GPU processes)
- **Iteration GPU-hours**: gpu_hours_total ≈ 11.7 (6.14 productive fixed batch + 5.35 wasted mkdssp-PATH-bug run + ~0.2 pilots/debug); gpu_ids witnessed {3,4,5} ⊆ {2,3,4,5}

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② main-experiment mechanism-harness fix (+ ⓪ C1 relabel, ⓪ C2 caveats) | C3 (C1, C2) | — | 3 | ~11.7 (6.14 productive) | 6.8 | almost |
| 2 | pure re-review (no back-edge; STOP) | — | — | 0 | 0 | 6.8 | almost |

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close within scope. These are non-blocking (no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE remains) but should be carried into the paper / a future verify pass.

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none (C3 repaired to INTEGRITY_ONLY).
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested under swaps)**:
  - **C1** — stage2_skip_reason: max_verify_claims_cap; main-experiment integrity WARN (relabeled ⓪). Upgrade: `/auto-verify C1 — resume: true`.
  - **C3** — stage2_skip_reason: max_verify_claims_cap; main-experiment integrity WARN (warn_source: experiment+mechanism — mechanism-audit residual WARN: α not in σ_proj units; locked α=8 at the capability-preserved edge, not a demonstrated interior plateau middle). Upgrade: `/auto-verify C3 — resume: true`. Swap-robustness (method/dataset/model) never evaluated for C3.
- **Legacy deferred claims**: none.
- **Recurring unresolved patterns (residual, non-blocking)**:
  - C3 specificity is *narrowly* sufficient: N=33 null (p=0.029 is the zero-exceedance extreme), single decisive dose, one predictor stack, modest effect (+0.096). Strengthen with more random directions, an intermediate dose (e.g. α=6) to show an interior plateau middle, and a structure-predictor swap.
  - Keep C2 tightly bounded to "proxy structure-control along the SS-assignment axis under one predictor stack" — do not drift to "structure control broadly demonstrated."
  - The ESMFold→OmegaFold predictor-swap axis remains untested for C2 (and would also strengthen C3).
- **Claim-reentry refusals**: none (no type-③ requested).
