# Refinement Report

**Date**: 2026-07-14
**Behavior-source**: given / **Mechanism**: discovery
**Final verdict**: READY

## What was refined

Given the `behavior_source: given` setting, `task.md`'s three sub-claims are the fixed anchor. Refinement therefore focused on the **verification methodology**, not the claim text.

### Refinement 1 — Location screen: full ACDC → attribution patching

- **Before**: an ACDC-style iterative pruning of the full computation graph — reference-quality but ~4 GPU-h alone on Mistral-7B.
- **After**: attribution patching (Syed et al. 2024, arXiv 2310.10348) as the cheap first pass → completeness/minimality re-verification only on the shortlist (Miller et al. 2024, arXiv 2410.13032).
- **Why**: compute — the shortlist size then bounds the cost of the actual path-patching runs (M2/M3). Attribution patching has comparable recall to ACDC per the source paper.

### Refinement 2 — Corruption strategy: zero-ablate → resample-ablate

- **Before**: (implicit default) zero-ablate the corrupted components.
- **After**: resample-ablation from a held-out `pool_resample` pool.
- **Why**: Best-Practices paper (arXiv 2309.16042) shows zero-ablate systematically misleads because it moves activations off-manifold. Resample-ablation from unrelated prompts keeps activations on-manifold.

### Refinement 3 — Metrics: single metric → three metrics reported side-by-side

- **Before**: report `logit_diff` alone (typical practice).
- **After**: report `logit_diff`, `prob_diff`, and `KL(clean || corrupt+patch)` for every intervention.
- **Why**: Best-Practices and Heimersheim & Nanda (arXiv 2404.15255) both warn that metric choice can flip results. Reporting all three is the accepted hedge (Gap G3 in the landscape).

### Refinement 4 — C2 modularity: positive-evidence → explicit dissociation + null control

- **Before**: "here are three sets of heads that look like fact/rule/answer heads" — positive-evidence style.
- **After**: (a) per-component role-assignment matrix S with per-row dominance ratio and per-column dissociation score, (b) label-shuffle null-hypothesis test with p-value.
- **Why**: Arithmetic-heuristics paper (Nikankin et al. 2024, arXiv 2410.21272) shows that at least one reasoning task in an LM is best explained by a sparse *heuristic bag*, NOT a modular circuit. The null must be explicitly refuted.

### Refinement 5 — Cross-family scope: match indices → schema recurrence only

- **Before**: (naive) look for the same head indices across Mistral and Gemma.
- **After**: report at the *schema* level — sparsity fraction, three-role dissociation, necessity/sufficiency recovery — not head indices.
- **Why**: cross-family head-index correspondence is neither expected nor testable; the schema question is the informative one.

### Refinement 6 — Gemma-2-27B: mandatory → contingent

- **Before**: (default) include Gemma-2-27B as a main verify swap.
- **After**: run only if budget headroom permits and gemma-2-27b downloads cleanly. Gemma-2-9B is the guaranteed cross-family swap.
- **Why**: 10-GPU-h budget with 7B main + 9B verify + 27B verify would need ~13 GPU-h at the same rigor. Task.md allows "use as needed, not necessarily all"; contingent triage honors that.

## What is intentionally NOT in the plan

- No SAE / transcoder / dictionary decomposition
- No training-checkpoint analysis
- No fine-tuning or editing
- No natural-language (non-synthetic) prompt experiments
- No head-index matching claim across families
- No claim about the reproduction target paper (arXiv 2411.04105) — it is not read; sub-claims come from `task.md` verbatim
