# Experiment Audit Report — Claim C2

**Date**: 2026-07-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C2 — Some identifiable internal component-kind of the DiT / MMDiT causally carries the transferred bias
**Linked milestones**: M1 (Location screen), M2 (Causal intervention)

## Overall Verdict: WARN

*C2's experimental process is methodologically sound in structure — the M1 Grassmann-overlap screen is well-reasoned and M2's forward-hook steering implementation is correct. The primary caveat is a scope reduction: the planned 8-seed × 5-intervention grid was reduced to 4 seed × 3 intervention = 12 runs, meaning the result is based on a 30% scope sample. The wording "some identifiable internal component" is directionally consistent with the evidence (block 50 is genuinely identified), but the causal claim (it "causally carries" the bias) was not established at this scope because the intervention effect is too small. The methodology is sound; the result is negative / under-power.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
The outcome metric is `P(banana) = banana_count / 160` using the task-specified gpt-5.4 judge at temperature=0.0 with the fixed 10-way prompt — the same instrument used for M0. No external reference dataset exists for this task (generated-image preference evaluation), and gpt-5.4 is the pre-registered measurement instrument defined in task.md. The Grassmann-overlap signal in M1 derives from the LoRA ΔW singular value structure — no GT label from another model is involved in the causal mechanism check. The forward-hook steering in M2 is applied in image-generation space, and outcome P(banana) is read off the judge, again symmetric across all conditions.
- **Evidence**: `src/mechanism/intervene_and_eval.py` imports and calls `judge_batch_parallel()` with `JUDGE_MODEL="gpt-5.4"`, `temperature=0.0`, same 10-way prompt as M0. `src/mechanism/location_screen.py` operates on torch `.pt` LoRA weight files — no external annotation.

### B. Score Normalization: PASS
`p_banana = banana_n / max(1, len(labels))` where `len(labels)` = 160 (fixed, prompt-count-determined, not model-dependent). Δ_ablate, Δ_random are simple arithmetic differences between experimental conditions sharing the same denominator. No normalization by the model's own max or mean.
- **Evidence**: `intervene_and_eval.py` line: `"p_banana": banana_n / max(1, len(labels))`. EXPERIMENT_RESULTS.md reports deltas as raw p_banana differences. No ratio involving a model-specific quantity.

### C. Result File Existence + Scope Reduction: WARN
Twelve of 12 scoped runs (4 seeds × 3 interventions) have `runs/M2_intervention/` artifacts with `verdict.json` and judge output. Numbers in EXPERIMENT_RESULTS.md (Δ_ablate_mean = -0.015, Δ_random_mean = -0.019, fluency 0.80–0.98) are consistent with the per-seed table reported in that file.

**Scope reduction caveat (WARN)**: the plan committed to `8 seeds × 5 interventions = 40 runs`. The realized grid was `4 seeds × 3 interventions = 12 runs` — a 70% reduction from the plan. The selected seeds {200, 202, 205, 207} are described as "spanning low/mid/high P(banana)" — a defensible representative subset. The selected interventions {ablate, amplify_x3, random_ablate} directly answer the three core C2 predicates (Δ_ablate drop, dose-response proxy via amplify_x3, and specificity via random_ablate). However, the dose-response monotone check requires amplify_x{2,3,4} — with only amplify_x3, the planned Spearman-rho ≥ 0.9 test cannot be computed (single-point dose curve). This is a documented scope caveat, not fabrication or methodology failure.
- **Evidence**: Tracker rows R039–R050 (only 12 of planned 40 rows); EXPERIMENT_RESULTS.md §Data actually used row for M2 explicitly states "scoped to fit 10 GPU-hour budget"; `runs/M2_intervention/dispatch.sh` shows 12 job entries.

### D. Dead Code Detection: PASS
`intervene_and_eval.py` — `parse_args()`, `_load_direction()`, `_estimate_sigma()`, `_hook()`, and `main()` are all on the active execution path via `if __name__ == "__main__"`. All metric fields (`p_banana`, `fluency`, `sigma_l`, `labels`) are computed and written. `location_screen.py` similarly writes `shortlist.json` with `sites`, `per_signal_topk`, `per_signal_heatmap_path`. The Grassmann-overlap computation and direction extraction are live code paths — confirmed by `runs/M1_location/shortlist.json` and `banana_direction.pt` existing on disk.
- **Evidence**: `runs/M1_location/shortlist.json`, `runs/M1_location/banana_direction.pt`, `runs/M2_intervention/` containing 12 verdict.json files.

### E. Scope Assessment: WARN (same caveat as Check C)
The C2 claim reads "some identifiable internal component-kind of the DiT / MMDiT causally carries the transferred bias." The plan's success criterion requires 5 predicates to pass on ≥6 of 8 seeds. The realized evidence covers only 4 seeds and omits the amplify_x{2,4} interventions needed for the full dose-response curve and the matched-random control at 8 seeds. The claim's minimum evidence bar (i) through (v) as written in the plan is not fully satisfiable from the 12 collected runs: in particular, the monotone dose-response check (predicate iii) requires ≥3 amplification points. This is a scope limitation, not an overclaim of interpretation — the result is reported as "not-supported [provisional — suspected under-power]", which accurately characterizes the evidential gap.
- **Evidence**: EXPERIMENT_RESULTS.md explicitly describes the scope reduction and labels C2 "weak-support / delocalized"; main-experiment-verdicts.json records verdict as "not-supported" with confidence "medium".

### F. Evaluation Type: task_specified_proxy
M1 evaluation uses LoRA ΔW singular-value structure (no external GT — the mechanism screen is fully internal). M2 evaluation uses the same gpt-5.4 task-specified vision judge as M0. Both are consistent with task.md's designated measurement instruments. Neither substitutes an undeclared proxy.
- **Evidence**: Task.md HARD constraint judge field; EXPERIMENT_PLAN.md M1 and M2 evaluation specifications; `location_screen.py` and `intervene_and_eval.py` imports.

## Action Items
- **Scope caveat (WARN)**: The 4-seed × 3-intervention result is scientifically interpretable but does not satisfy the plan's original ≥6/8 seed majority criterion. The dose-response curve (requiring amplify_x{2,3,4}) is a single-point measurement here. To resolve: run the remaining 4 seeds × remaining 2 interventions (amplify_x2, amplify_x4, and seeds 201, 203, 204, 206) in a follow-up round. This is a scope gap, not a methodology failure — the 12 completed runs correctly motivate the delocalized interpretation.
- **Forward-hook scope (informational)**: The forward hook is registered on `transformer_blocks[50]` (the single block identified by M1). The delocalized result implies the carrier direction spans multiple blocks; a multi-block simultaneous hook would be the natural follow-up intervention.
