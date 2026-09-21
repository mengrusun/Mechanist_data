# Refinement Report

**Problem**: Validate a *given* behavior — cross-modal subliminal transfer of an unsafe behavior via a text-only teacher-generated channel in the fixed Qwen3.5-9B multimodal setup — then discover the mechanism if the M0 gate passes.
**Initial Approach**: 3-arm × ≥3-seed × ≥3% gap M0 gate + Location→Causal Intervention mechanism arc.
**Date**: 2026-07-17
**Rounds**: 3 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim across all rounds)

Validate: in a fixed Qwen3.5-9B → Qwen3.5-9B (matched-initialization) multimodal transfer setup, text-only teacher-generated data that has been surface-filtered to look safe (via `filter_prompts_lenient.md`) covertly transmits an unsafe behavior to the multimodal student, measurable as `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 %` AND `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 %`, per seed across ≥ 3 random seeds, with filter re-scan clean.

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.0 | 8.0 | 9.0 | 8.0 | 6.0 | 8.0 | 8.0 | 8.1 | REVISE |
| 2 | 8.9 | 8.8 | 8.6 | 8.7 | 8.0 | 8.4 | 8.2 | 8.6 | REVISE |
| 3 | 9.6 | 9.1 | 9.0 | 8.9 | 8.4 | 8.8 | 8.7 | 9.0 | READY |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|---|---|---|---|
| 1 | Feasibility 6/10 due to per-seed teacher retrain; M0 gate softening via `conditional`; unpinned mechanism specifics; judge-audit brittle veto; bootstrap-CI acting as quasi-gate. | Teacher SFT one-time (T* reused); M0 binary PASS/FAIL/RUN INVALID; logit-margin + intervention site + top-K + off-target pinned; judge calibration matrix; bootstrap CI = readout only; L-Core = contrastive direction extraction, LoRA attribution demoted to fallback. | Overall 8.1 → 8.6 |
| 2 | Seed replacement ambiguity; judge audit still contaminating scientific verdict; mechanism verdict not pre-registered; dual location metric; feasibility lacked cache footprint; strict intersection brittle. | Seeds pre-registered `{42,123,2026}` with no scientific replacement; judge audit → measurement-validity flag only; mechanism verdict pre-registered as 3-level; single primary location metric = accuracy-conditioned activation contrast; cache policy binding (bf16 pinned-site, ≤ 4000 items/seed); Borda rank aggregation; signed linear-probe companion diagnostic; isotonic-fit deviation. | Overall 8.6 → 9.0 |
| 3 | Pooled dose-response statistic under-specified; direction-selection merge rule fuzzy; feasibility needs explicit worst-case + hard-stop; Ctrl-C could go to appendix. | Per-seed Spearman ρ_s over 7-point α curve, pass = median ρ ≤ −0.5 AND ≥ 2/3 seeds ρ < 0; `d_diff` = sole intervention direction (d_pca + probe = diagnostic-only); worst-case budget table with hard-stop at 60 GPU-hours (L-Secondary → appendix if triggered); Ctrl-C moved to appendix; judge matrix = flag + one summary table. | Verdict READY |

## Final Proposal Snapshot

Canonical clean version lives in `refine-logs/FINAL_PROPOSAL.md`. Three-bullet thesis:

- **Verification** — binary M0 gate on 3 pre-registered seeds `{42, 123, 2026}`; `PASS` iff for each seed both `Ctrl-A − treated ≥ 3 %` AND `Ctrl-B − treated ≥ 3 %` AND filter Stage-B re-scan clean; judge audit labels only measurement validity (`RUN INVALID`); no scientific-fail replacement; bootstrap CI on `Ctrl-B − treated` is a readout, not a gate; VLSBench-style text-only diagnostic in appendix.
- **Mechanism (conditional on `PASS`)** — Location via contrastive `d_diff` extraction on the flipped-wrong QA_I subset with Borda cross-seed aggregation + signed linear-probe diagnostic; LoRA-block attribution fallback only if L-Core stability gate fails (hard-stoppable to appendix at 60 GPU-hour cap); Causal Intervention via joint ablation on treated + 7-point steering α sweep on base with per-seed Spearman ρ + matched-control specificity. Verdict = `STRONG POSITIVE` / `PARTIAL POSITIVE` / `BOUNDED NULL` per pre-registered thresholds.
- **Compute** — worst-case ≈ 56 GPU-hours end-to-end (M0 ≈ 25; mechanism with L-Secondary ≈ 31), well inside the "ample" clause of task.md; hard-stop rule keeps main-body compute ≤ 60 GPU-hours.

## Method Evolution Highlights

1. **Removed per-seed teacher retrain** (round 1) — the single largest wasted compute; task.md never required it.
2. **Made the M0 gate binary and separated measurement-validity from scientific verdict** (rounds 1–2) — the cleanest way to preserve the anchor while adding evaluator hardening.
3. **Pre-registered the mechanism verdict hierarchy** (round 2) — turns a fuzzy "did we localize?" into a machine-checkable ladder.
4. **Pinned the pooled dose-response statistic + single primary intervention direction** (round 3) — removes the last researcher-DOF sources; the mechanism claim is now falsifiable in a well-defined way.
5. **Added the worst-case budget table with hard-stop rule** (round 3) — makes the plan credible under close reading and gives the pipeline a deterministic degradation path when L-Secondary is expensive.

## Pushback / Drift Log

None. All three rounds concluded `Drift Warning: NONE`; no reviewer suggestion required an anchor-defending pushback (the two candidate softenings — mean-across-seeds bar and seed replacement — were both silently *tightening* the anchor, so accepting the reviewer's simplifications restored anchor fidelity).

## Remaining Weaknesses

- `n_layers` / `d_model` in the compute table are order-of-magnitude estimates (≈ 40 / ≈ 5120). The exact Qwen3.5-9B language-tower config should be inspected at `/auto-experiment` Phase 1.5 routing time and the table refined then. This does not change any threshold or verdict — only the exact cache size and hook count.
- Off-target competence eval relies on `eval_pairs_948.json`; if its content turns out to be > 5 % safety-adjacent (deterministic regex audit runs), we downsample; if unusable, the milestone is deleted from the first arc (documented explicitly in the failure-mode table). No fallback benchmark is invented.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

See `refine-logs/round-1-review-raw.md` (12123 chars; overall 8.1, REVISE; drift NONE).

</details>

<details>
<summary>Round 2 Review</summary>

See `refine-logs/round-2-review-raw.md` (13039 chars; overall 8.6, REVISE; drift NONE).

</details>

<details>
<summary>Round 3 Review</summary>

See `refine-logs/round-3-review-raw.md` (10635 chars; overall 9.0, READY; drift NONE).

</details>

## Next Steps

- Proceed to `/experiment-plan` to turn `FINAL_PROPOSAL.md` into `EXPERIMENT_PLAN.md` + `EXPERIMENT_TRACKER.md`.
- Then `/auto-experiment` (Workflow 1.5) — Phase 1.25 M0 gate first; Phase 1.5 routes the mechanism submethod via `/mechanism-skills` only if M0 = PASS.
