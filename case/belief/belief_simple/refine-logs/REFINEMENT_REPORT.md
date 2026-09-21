# Refinement Report — Reasoning for R1-R5 Gap Resolutions

**Date**: 2026-07-22
**Scope**: Reproduction of "Sensitivity Meets Sparsity" (Chen et al., 2025) on Pythia-{410m, 1b, 2.8b}.
**Style**: For a strict-fidelity reproduction, the only refinement decisions are on ambiguous points in the recipe. This report documents the reasoning behind each pin.

---

## R1 — pythia-410m above-chance gate

**Ambiguity**: task.md's Claim 2 says: "For models that perform above chance on the target task, identify candidate heads using the third-person subset (person in {James, Mary}) and test them with zero-ablation." — but does not define the "above-chance" threshold.

**Reasoning**:
- The behavioural metric is a >-comparison of gold vs distractor log-probs → binary → chance = 0.5.
- A single-point accuracy above 0.5 is not sufficient evidence given the moderate sample sizes (n=227 for `world_knowledge`, n=681 for the two belief tasks; the sample-size-scaled standard error at p=0.5 is ~0.033 for n=227 and ~0.019 for n=681).
- Standard practice in the ToM-in-LMs line (Kosinski, Ullman, Chen et al., etc.) is to require the lower 95% CI to also exceed chance.

**Pin**: `acc > 0.5 AND lower Wilson 95% CI > 0.5`. Wilson interval used because it has better small-sample coverage than Wald.

**Consequence**: pythia-410m is expected to be borderline on `attributed_belief` (false-belief tasks are hard for small LMs — Ullman 2023 showed even much larger models struggle). If 410m fails the gate for a belief target, M2 skips that (model, target) config and reports `not_applicable` — the plan does NOT lower the gate to force 410m through.

---

## R2 — "Smallest head set" search discipline

**Ambiguity**: task.md's Claim 2 method says "search for the smallest head set that satisfies the causal, specificity, baseline, and PPL criteria" — but does not specify the search algorithm. Greedy-add? Greedy-remove? Branch-and-bound? ACDC-style graph pruning?

**Reasoning**:
- The candidate-head ranking from the Fisher mask is *the* signal task.md tells us to use. A greedy-add over this ranking is the natural, deterministic, defensible search.
- Wang et al. 2022 (IOI) and Chen et al. 2025 both use greedy candidate-add methodology; branch-and-bound is intractable for the number of candidate heads (up to `layers × heads` — for pythia-2.8b that's ~1024 heads).
- Greedy-add alone can find a *local* minimum: adding heads in ranking order may over-shoot the "smallest" set if two mid-ranked heads together satisfy the criteria and a low-ranked head is unnecessary. So we add a **greedy-remove sanity check**: iterate through the found `S⁺`; attempt one-shot removals; if any removal still satisfies the criteria, take the smaller set.
- To bound search cost we cap `|S| ≤ 30`. This is well beyond the ~1-10 heads Chen et al. and IOI-line report — a set requiring more than 30 heads is already outside the "sparse localization" regime. If the cap is hit, honestly report `not_localized` rather than raising the cap.

**Pin**: Greedy-add ranked-order → first passing `S⁺` → greedy-remove sanity check → fixed point → cap 30 heads. All deterministic (no random tie-breaks; ties broken by aggregated Fisher magnitude descending).

**Rejected alternatives**:
- ACDC (Conmy et al. 2023): Adds substantial machinery (graph pruning over edges, not heads); would deviate from the paper's per-head design.
- Branch-and-bound: intractable at pythia-1b/2.8b scale.
- Random-restart greedy: adds non-determinism; not needed given the deterministic ranking.

---

## R3 — PPL corpus scope

**Ambiguity**: task.md's Claim 2 4th criterion is "PPL after ablation is no more than 1.05 × the clean PPL", with the pretraining corpus location fixed at `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/`. Sample-size and seed are unspecified.

**Reasoning**:
- The 4-criteria acceptance test is a *paired* comparison across many runs (1 clean + 1 main ablation + 20 random-head controls + 20 random-mask controls = 42 PPL numbers per (model, target)). Any variation in the PPL sample across runs would inject noise unrelated to the intervention — paired comparison requires the same sample everywhere.
- 1M tokens is a well-established Pile-PPL sample size (Chen et al.'s and Chinchilla-line papers use ~1M-token PPL windows). Per-token PPL stability at 1M tokens is within ±0.01 (standard error over independent samples), which is much tighter than the 1.05× threshold demands.

**Pin**: `PPL_TOKENS = 1,048,576` tokens. Sample constructed once by streaming the pre-shuffled Pile shards, tokenizing with the model's own tokenizer, truncating to `PPL_TOKENS`, and caching to `refine-logs/artifacts/ppl_sample.pt`. Every PPL call (clean, ablated, control, formation-window causal, controller general-LM check) reads from this cache.

**Rejected alternatives**:
- Fresh random sample per run: adds noise → false negatives on the 1.05× criterion.
- Full-corpus PPL: computationally infeasible (~300B tokens; 1000× more expensive than needed).
- Per-model tokenizer sample: unnecessary — all three Pythia models share the same tokenizer, so the cached tensor is model-agnostic.

---

## R4 — Claim-3 checkpoint schedule

**Ambiguity**: task.md's Claim 3 says "measure behavioral evaluation at different pythia-1b training steps" — but does not specify which subset of the 154 available checkpoints to use.

**Reasoning**:
- Pythia intentionally provides 154 checkpoints on a mixed schedule (11 early log-spaced + 143 later linear-spaced) precisely because both regimes matter: the early log-spaced steps catch rapid emergence phenomena (induction heads emerge around step 512-4000; Olsson et al. 2022, Singh et al. 2024); the later linear-spaced steps let us track consolidation and post-emergence dynamics.
- Sub-sampling the schedule risks missing the emergence step (which is exactly what "formation window" measures). Given all 154 checkpoints are already on disk, using all of them is the honest choice.
- The cost is bounded: 154 checkpoints × 9 evals ≈ 47h on pythia-1b — well within reproduction budget.

**Pin**: Use the full 154-checkpoint native Pythia schedule (enumerated in EXPERIMENT_PLAN.md M3 grid). Do NOT sub-sample.

**Rejected alternatives**:
- Sub-sample every 5th checkpoint: risks missing emergence at a skipped step.
- Only later linear checkpoints (skip 0-512): would miss the earliest emergence signals.
- Only early log checkpoints (skip after 10000): would miss consolidation.

---

## R5 — Claim-4 amplification magnitude

**Ambiguity**: task.md's Claim 4 describes head amplification but does not specify the amplification coefficient α.

**Reasoning**:
- Activation-steering literature (Turner et al. 2023, Panickssery et al. 2024) treats α as a critical hyperparameter — too small has no effect; too large destroys general LM ability (PPL blowup) and can over-shoot into off-target frames.
- The typical range for head-level amplification in mechanistic-interpretability steering (Bortoletto et al. 2024) is `α ∈ {1.5, 2, 3, 4, 6}`. We include `α = 1.0` (no amplification) as an explicit baseline in the grid.
- Two independent α's are needed because H*_personal and H*_attributed may have different sensitivities. 6 × 6 = 36 configs is tractable (~3.6h per model on a single GPU).
- Selection on the belief_core val split (the 20% held out from classifier training) — evaluated on the OOD holdout only after α selection is frozen. This is standard hyperparameter-selection discipline and avoids double-dipping.
- The soft guardrail `Δ_wk ≤ 0.05` prevents α selection from destroying factual knowledge on val (which would generalize to OOD destruction). Relaxes to 0.10 if no combination clears 0.05; reports failure if even 0.10 is not clearable.

**Pin**: `α_personal ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`, `α_attributed ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` (36 configs). Selection by `argmax net_improvement subject to Δ_wk ≤ 0.05` on val. Selected pair frozen for OOD evaluation.

**Rejected alternatives**:
- Learn α by gradient descent: adds a training loop; deviates from the "lightweight controller" phrasing in task.md; overkill for 2 scalars.
- Fix α at a single value: no evidence the paper uses one specific value; would arbitrarily pick from the range.
- Broader grid (e.g., α ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0}): 100 configs = ~10h per model, unnecessary given the coarser 6-value grid captures the range well.

---

## Summary

Each pin above is a *deterministic implementation choice on a specification gap*, not a *design change to the method*. The reproduction stays faithful to task.md's four claims and their prescribed methodology; the pins guarantee that anyone re-running this project follows exactly the same recipe.
