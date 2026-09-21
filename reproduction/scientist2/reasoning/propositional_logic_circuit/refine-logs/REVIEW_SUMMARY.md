# Review Summary — Sparse Modular Circuit Reproduction

**Date**: 2026-07-14
**Reviewer role (internal, this pass)**: senior mechanistic-interpretability reviewer (NeurIPS / ICML standard). External-LLM review is deferred to `/research-review` when invoked separately — the current pass is the self-review baked into `/research-refine-pipeline`.

## Scores

| Criterion | Score / 10 | Notes |
|---|---|---|
| **Problem clarity** | 9 | Three sub-claims are crisp and falsifiable, verbatim from `task.md`. |
| **Methodological fit** | 9 | Uses the accepted IOI → circuit-hypothesis pipeline (Wang 2022, Conmy 2023, Syed 2024, Miller 2024). |
| **Novelty vs reproduction** | 5 | This is a *reproduction with cross-family generalisation*. Novelty comes from (a) applying the schema-recurrence test at 7B/9B parity of rigor, (b) explicit dissociation + null control against the bag-of-heuristics account. Not a novelty-driven paper. |
| **Falsifiability** | 9 | Each of the three claims has an explicit success threshold with reported metrics. Negative results are publishable. |
| **Compute-budget realism** | 8 | 8.7 GPU-h fits under 10 with 1.3-h headroom. Gemma-2-27B contingent — sensible triage. |
| **Baselines and controls** | 9 | Matched-control patches for specificity; null-shuffle for modularity; all-three-metrics reporting for metric-sensitivity hedge. |
| **Overall** | **8.2** | READY. |

## Weaknesses acknowledged

1. **Metric selection is still an open sub-question.** Choosing `logit_diff` as the primary metric vs `prob_diff` matters — Best-Practices paper shows they can disagree. Mitigation: report all three.
2. **Prompt-specificity of circuits** (Franco & Crovella 2025) means the schema could hold within a template family but not generalise to natural-language prompts. Mitigation: (a) cross-cell stability check with three lexicon variants including symbolic and alt-noun lexicons; (b) the reproduction *scope* explicitly stays inside the synthetic template — natural-language generalisation is out of scope for this pass.
3. **Sufficiency reinsertion at 7B is subtle** — corruption strategy matters. Mitigation: resample from a large pool (per Best-Practices, arXiv 2309.16042); report distribution over resample seeds (≥ 5) with std check.
4. **Cross-family head-index matching is impossible.** We report only *schema-level* recurrence (sparsity fraction, three-role structure, dissociation score). Explicitly do NOT claim head-index correspondence.
5. **Mistral-7B-v0.1 local symlink is broken.** Fallback path documented (Mistral-7B-Instruct-v0.1); download-if-needed at M0.setup.

## Complexity intentionally rejected (final list)

- SAE / dictionary decomposition (outside GPU budget)
- Formation tracing (out of scope — final checkpoint only)
- Tuning & Editing (no editing objective in reproduction)
- Decision Auditing (orthogonal to the three sub-claims)
- Gemma-2-27B main-track inclusion (contingent instead)
- Full ACDC iterative pruning (replaced with faster attribution-patching cheap screen)
- New synthetic prompt families beyond the parameterised template
