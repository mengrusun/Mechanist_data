# Final Proposal — SAGE Reproduction: Unified Testing Approach for the Four Captured Claims

**Date**: 2026-07-14
**Behavior-source**: given
**Mechanism**: discovery (`chosen_mechanism` is NOT stamped — the experiment stage's Phase 1.5 routes the family)
**Resource fidelity**: `resource_fidelity: normal` (cost-aware; this is the `given` + `discovery` combination, not the `given` + `given` reproduction combo)
**Mechanism strategy** (from `/mechanism-explore`):
```yaml
mechanism_strategy:
  directions: [Unit Interpretation]                    # in execution order — single direction; SAGE is a Unit-Interpretation pipeline (Direction 5)
  rejected:
    - Location — the target unit (an SAE feature id) is already given, so location is trivial (the SAE encoder does it); nothing to search
    - Causal Intervention — SAGE's *claim* is about explanation quality (generative + predictive accuracy), not about whether the feature causally drives a downstream behavior; intervention would test a different claim
    - Tuning & Editing — no capability-tuning goal in task.md; the pipeline outputs labels, not weight edits
    - Formation Tracing — no training-time claim; task.md is entirely about inference-time explanation of an already-trained model
    - Decision Auditing — no downstream decision to audit; the metric is explanation faithfulness to the SAE feature itself
  note: SAGE decodes what an internal unit (an SAE feature) means via an iterative model-explains-model loop. It is the auto-interpretation variant of Direction 5.
```

## Problem Anchor (frozen)

The four claims C1-C4 in `idea-stage/IDEA_REPORT.md`, taken verbatim from `task.md`. These do NOT move; refinement only tightens the *testing method* around them.

## Method Thesis (of the pipeline being reproduced — not modified)

An SAE-feature explanation is treated as a hypothesis about the feature's activating conditions. Rather than accepting the first natural-language description the explainer LLM emits from top-activating snippets (the Neuronpedia / Bills-2023 / Paulo-2024 paradigm), SAGE runs an **agentic loop** with four separated roles — all played by GPT-5:

- **Explainer** — proposes several candidate natural-language explanations for feature f, each covering a distinct hypothesis about what f responds to.
- **Designer** — writes targeted probe inputs to *discriminate between* the candidate explanations: text that any given candidate predicts should strongly activate f (or specifically should not), across a range of activation levels rather than only top-k.
- **Analyzer** — pushes each probe through the target LLM + SAE, reads the empirical activation of f, and scores each candidate against the observed activations (empirical activation feedback).
- **Reviewer** — decides per round whether to accept, reject, or revise each candidate; loop halts when a stopping criterion is met (agreement across candidates, activation-fit above threshold, or maximum-rounds cap).

The output is one (or several, if multiple candidates survive with distinct concept coverage) natural-language explanation(s) per feature id. This *pipeline* is what is being reproduced; whether it beats a Neuronpedia one-shot baseline on the same feature ids on both metrics is what is being verified.

## Dominant Testing-Approach Design Choices

The claims' scope is fixed. The testing method has four load-bearing choices whose defaults are made explicit here so the experiment stage does not silently drift:

1. **Same-feature-id paired design.** SAGE and Neuronpedia are evaluated on identical feature ids in identical SAE checkpoints. Paulo & Belrose 2025 (arXiv 2501.16615) shows SAE features are seed-dependent — cross-checkpoint comparison would be scientifically invalid. Under `resource_fidelity: normal` this is **still** enforced, because relaxing it would falsify the claim, not save cost. Consequence: statistics are paired (paired bootstrap / paired Wilcoxon), not unpaired.
2. **Layer stratification.** Features are sampled equally from early, mid, and late residual-stream layers. C3 requires this — pooling across layers alone would leave the depth question unanswered.
3. **Same-backbone Neuronpedia baseline sanity control.** Neuronpedia's public explanations were produced by GPT-4-class models; SAGE uses GPT-5. To disentangle *pipeline* from *backbone*, milestone M0.5 (methodological gate — NOT phenomenon-validation) also runs a **single-pass GPT-5 explainer with a matched prompt** against Neuronpedia's public explanations on the same features. If the single-pass-GPT-5 baseline already beats Neuronpedia, C1-C4's target contrast is redefined against the single-pass-GPT-5 control for interpretation purposes (headline vs. Neuronpedia is still reported, but the *pipeline attribution* rests on beating the matched-backbone control). This does NOT change the claim — it makes attribution honest.
4. **Independent judge / scorer LLM.** The probe-writer for C1 (generative accuracy) and the activation-predictor scorer for C2 (predictive accuracy) must NOT be one of SAGE's four internal roles for the same feature — otherwise the pipeline would grade its own probes. Both are separate GPT-5 sessions with independent prompts and no access to SAGE's internal state; the same judge / scorer is used for the Neuronpedia baseline and the single-pass-GPT-5 control (paired design across methods).

## Elegance / Simplicity Statement

Nothing beyond `task.md` is added to the *claim*. All refinement is at the level of *how* to test: (i) same-feature-id paired design, (ii) layer stratification, (iii) matched-backbone control, (iv) independent judge/scorer. The metric definitions (generative + predictive accuracy) are directly `task.md`'s; the statistical protocol (paired bootstrap / paired Wilcoxon with Bonferroni across depth × metric) is the standard applied to paired feature-level scoring; no new definition of "accuracy" is introduced.

## Frontier Awareness

The proposal explicitly connects to the closest prior evaluation harness — Paulo et al. 2024 (arXiv 2410.13928)'s detection + intervention scoring. **Predictive accuracy** maps to Paulo's detection score; **generative accuracy** maps to Paulo's intervention score adapted to a "probe-writing → activation-check" protocol rather than a weight-clamp intervention. Reusing this harness where possible reduces the risk that positive results reflect an idiosyncratic evaluation implementation.

## Risks / Open Questions Deferred to the Experiment Stage

- **R1** — Neuronpedia's activation corpus for `gemmascope-res-16k` may not expose a labeled train/held-out split; the experiment stage may need to *induce* the split (e.g. random 80/20 per feature, seed-locked) so the "held-out text" in C2 is well-defined. The plan flags this in M0.5.
- **R2** — GPU-budget confinement (10 GPU-hours) is generous for a labeling pipeline (all target LLM forward passes are read-only through SAE hooks, no gradients); the plan should be sized to the full 300-feature-per-model plan and downscale only if a pilot shows the budget is genuinely insufficient.
- **R3** — The 10-hour budget across main + verify + iteration constrains M2 to *one* verify LLM+SAE pair inside main experiment; the second pair is deferred to `/auto-verify`.
- **R4** — The mechanism-family submethod is not committed at claim time (`MECHANISM=discovery`); the experiment stage's Phase 1.5 will bind `n_pairs`, `sites`, effect `metric`, and `gpu_hours` per milestone. Those fields are tagged `method_sensitive` in the plan.

## Next Steps

Handed off to `refine-logs/EXPERIMENT_PLAN.md` (this directory) → `/mechanism-skills` (routing) → `/auto-experiment` (implementation + deploy) → `/auto-verify` (stress-test) → `/auto-iteration-loop` (iterate if needed).
