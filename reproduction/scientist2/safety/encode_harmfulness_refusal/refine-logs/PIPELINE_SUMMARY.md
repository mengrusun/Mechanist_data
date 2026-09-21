# Pipeline Summary

**Problem**: Verify the five claims in `task.md` about the mechanistic separation of harmfulness perception and refusal execution as two dissociable linear directions in the residual stream of Llama-3-8B-Instruct on AdvBench.
**Final Method Thesis**: Extract h (harmfulness) at t_final-instr and r (refusal) at t_post-instr via difference-in-means on Llama-3-8B-Instruct residual-stream activations from paired (AdvBench-harmful, matched Alpaca-benign) prompts; verify Claims 1–5 through a focused pipeline of correlational probes (M1, M2), causal additive steering + specificity (M3), paired-attack projection deltas across GCG + PAP (M4), and a head-to-head classifier comparison against Llama Guard 3 8B at matched FPR (M5).
**Final Verdict**: READY
**Date**: 2026-07-15

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Faithfully-captured claims + literature landscape: `idea-stage/IDEA_REPORT.md`, `idea-stage/LANDSCAPE.md`, `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot
- **Dominant contribution**: claim-driven verification of the two-direction decomposition on Llama-3-8B-Instruct that localises h and r at two specific token positions, dissociates them causally with additive steering + specificity, replicates the refusal-suppressed / harmfulness-preserved jailbreak signature across two attack families (GCG + PAP), and puts a lightweight harmfulness-direction probe head-to-head with Llama Guard 3 8B at matched FPR.
- **Optional supporting contribution**: cross-model / cross-dataset generalisation is enumerated as verify-stage variants and consumed by `/auto-verify`, not the main-experiment budget.
- **Explicitly rejected complexity**: SAE feature-dictionary decomposition, formation tracing, per-decision auditing.

## Must-Prove Claims (verbatim from `task.md § Claim`; 1:1 in `idea-stage/IDEA_REPORT.md`)
- **Claim 1** — two distinct, approximately-linear, independently-recoverable directions (h, r) in Llama-3-8B-Instruct's residual stream.
- **Claim 2** — position dissociation: h at t_final-instr, r at t_post-instr (position crossover).
- **Claim 3** — causal dissociation via additive steering with monotone dose-response, off-target null-band, and specificity controls.
- **Claim 4** — refusal-suppressed / harmfulness-preserved signature on successful jailbreaks across GCG + PAP; hidden-state h-probe picks it up.
- **Claim 5** — harmfulness-direction probe ≥ Llama Guard 3 8B − ε at matched FPR, at ≤ 5% per-query compute.

## First Runs to Launch
1. **R001 (M-prep)** — Extract Llama-3-8B-Instruct residual-stream activations across every layer × candidate position on AdvBench + matched Alpaca benign; compute difference-in-means h and r; layerwise probe AUROC. (~1 GPU-h)
2. **R002 (M1)** — Run Claim 1's three sub-tests on cached activations: AUROC of h and r at their best (layer, position); cosine(h, r) w/ split-half reference; shuffled-refusal independence sanity. (~0.2 GPU-h)
3. **R003 (M2)** — Score AUROC per (position × attribute) on the position ladder around {t_final-instr, t_post-instr}; compute the crossover-Δ for Claim 2. (~0.2 GPU-h)

## Main Risks
- **Risk**: Low ASR of GCG on Llama-3-8B-Instruct → M4's per-family successful subset too small for reliable Δ. **Mitigation**: use published transferable GCG suffixes; expand behaviour set from 100 to 200 if per-family ASR < 20%.
- **Risk**: Best-layer × best-position from M-prep is not the true optimum. **Mitigation**: fine-grained re-sweep in the top-3 candidate layers before declaring C1/C2 failure.
- **Risk**: Llama Guard 3 8B threshold choice biases the M5 head-to-head. **Mitigation**: report AUROC (threshold-free) as primary; F1@FPR5% as matched-operating-point secondary.

## Next Action
- Proceed to `/mechanism-skills` (Workflow 1.25) to bind a concrete family/submethod for the intervention layer of M3 (the `method_sensitive` fields will be re-bound at routing time), then `/auto-experiment` (Workflow 1.5) to implement and run M-prep → M5 within the 10 GPU-h budget.
- Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
