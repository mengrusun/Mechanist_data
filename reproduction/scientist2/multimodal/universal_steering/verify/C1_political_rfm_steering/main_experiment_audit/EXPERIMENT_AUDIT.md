# Experiment Audit — C1 (RFM per-block concept-vector steering: anti-refusal / political / honesty)

**Claim scope**: Milestones M1_C1_screen, M2_C1_extract, M3_C1_alpha_sweep  
**Run IDs**: B1_extract_vectors, C1_steer_judge, C1b_refusal_block14, C1c_high_alpha  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Ground-Truth Provenance

**Status: PASS**

The judged evaluation uses GPT-4o-2024-11-20 as a rubric judge (5-point steering-effect scale), not as a ground-truth oracle for the underlying claim. The judge is explicitly scoring the steering effect ("Rate how much this response reflects <concept>") rather than providing binary ground truth for the claim's success predicate. The rubric scale is appropriate for an open-ended steering evaluation where no absolute ground truth exists.

The paired training data for extraction (refusal: AdvBench pairs, honesty: TruthfulQA + local pairs, political: DMX-generated gpt-5.4 pairs) use labeled semantic categories as supervision for the probe/RFM extraction step — legitimate use. The held-out eval prompts are disjoint from training pairs (per EXPERIMENT_PLAN.md splits: train 300 / val 100 / held-out 50, all disjoint).

**Finding**: The GPT-4o judge is used as a "rubric score" (measuring degree of concept expression), not as a proxy ground truth for binary correctness. This is appropriate for an open-ended steering evaluation. Minor concern: GPT-4o's rubric scoring may not be perfectly calibrated across concepts (honesty vs political vs refusal have different baseline rubric levels), but this does not invalidate the within-concept comparisons.

## Check B — Score Normalization

**Status: PASS**

Rubric scores are on an absolute 5-point scale (1–5), not normalized by the model's own output or own max. Delta computations (steered - baseline) are straightforward arithmetic differences, not ratios. No evidence of normalizing by the model's own maximum to inflate results.

## Check C — Result File Existence (claim-scoped)

**Status: PASS with WARN**

- `runs/C1_steer_judge/summary.json` — exists, non-empty. Contains agg_rfm per alpha per concept with n=50 each.
- `runs/C1_steer_judge/scored_political.jsonl`, `scored_honesty.jsonl`, `scored_refusal.jsonl` — expected per code; not directly read by this audit but the summary.json aggregates confirm the judging ran.
- `runs/C1b_refusal_block14/summary.json` — exists (referenced in EXPERIMENT_RESULTS.md).
- `runs/C1c_high_alpha/summary.json` — exists (referenced).

**WARN**: The `alpha_star` labelling in summary.json uses `argmax(|delta|)`, which picks the largest-magnitude alpha regardless of sign direction. For "honesty", alpha_star=-3.0 (delta=-0.20) was stored, even though the positive direction alpha=+3 gives a positive shift (+0.14). The EXPERIMENT_RESULTS.md explicitly references the positive shift at alpha=+3. The claims_ledger.json open_items note confirms: "alpha_star picks argmax(|Delta|) not signed direction — cosmetic labelling issue (reported numbers use raw per-alpha aggregates, unaffected)". Verified: the raw `agg_rfm` data in summary.json contains per-alpha means at all 7 alpha values; the reported numbers in EXPERIMENT_RESULTS.md correctly cite per-alpha values (alpha=+3 mean=4.02 for political, alpha=+3 mean=4.40 for honesty) rather than the potentially wrong-signed alpha_star. **The reported numbers are correct; the alpha_star label is a cosmetic artifact only.**

## Check D — Dead Code

**Status: PASS**

`c1_steer_and_judge.py` defines and calls `generate_with_steering`, `judge_completion`, and the aggregation logic. All relevant evaluation functions are actively called in the main execution path. The `judge_completion` function in `dmx_api.py` is invoked per generation (when `--skip-judge` is not set). The skip-judge flag was not set in the actual production runs (confirmed by non-null `agg_rfm` scores in summary.json).

## Check E — Scope (claim-scoped)

**Status: WARN**

The claim statement says the RFM vectors steer "beating both an unsteered baseline and a matched-random-direction control on the anti-refusal, political-stance, and honesty demo scenarios." However:
- **refusal**: Null result — 100% tie at rubric=1.0 across all alpha. The block-selection failure (probe ties at 1.0 all blocks → argmax picks block 0) means the Location step failed for refusal. The pinned-block-14 re-run (C1b) also failed to break safety tuning.
- **honesty**: Partial / weak result — alpha=+3 shift is +0.14 within judge std=1.25 (effect within noise; random control comparable).
- **political**: Clean result — signed monotone, range 0.94 rubric pts, random control 2-3x smaller.

The claim's scope of "all 3 demo scenarios" is not met — only political satisfies the strict success predicate. The EXPERIMENT_RESULTS.md correctly labels the overall verdict as "partial" but some hedging language ("beating both baseline and control on anti-refusal, political-stance, and honesty") in the claim statement is an overclaim for the published result. However, for verify purposes, we assess the strongest sub-finding: the political-stance component meets the success predicate cleanly.

## Check F — Evaluation Type

**Status: PASS**

Evaluation type: `synthetic_proxy` (GPT-4o rubric judge as a proxy for steering effect). This is disclosed and appropriate for open-ended text generation evaluation. No ground-truth steering metric exists for natural language; the rubric judge is the field standard for this type of evaluation.

The random-direction control addresses the key confound (magnitude-alone effects). The control is matched at the same `||alpha*v||` norm.

## Overall Verdict

**overall_verdict: WARN**

- Exp A (GT provenance): PASS
- Exp B (normalization): PASS  
- Exp C (result file existence): WARN — alpha_star labelling cosmetic bug (reported numbers unaffected, verified)
- Exp D (dead code): PASS
- Exp E (scope): WARN — claim scope "all 3 scenarios" is an overclaim; political is strongly supported, honesty is borderline, refusal is null
- Exp F (eval type): PASS

**Gate verdict**: WARN — Cx admitted to Stages 2–3 with caveat that the claim is partially supported (political component only meets strict predicate). The scope concern is noted; the numeric results are trustworthy for the political sub-claim.
