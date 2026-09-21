# Pipeline Summary

**Problem**: Faithfully reproduce the five claims in `task.md` about SAE-decomposed ESM-2-650M residual-stream features aligned with Swiss-Prot biological concepts, with mechanism-strategy shaping (Unit Interpretation → Decision Auditing → Causal Intervention) and full specificity controls at every claim.
**Final Method Thesis**: Reproduce all five claims under a single unified evaluation harness that uses the fixed pretrained InterPLM SAEs at six ESM-2-650M layers, computes SAE-vs-neuron alignment against Swiss-Prot per-residue annotations under an identical F1 protocol, adds three matched-capacity control arms (random-rotation / PCA / shuffled-SAE) for the superposition claim, runs an external LLM auto-interpreter with mandatory low-activation and random-feature baselines, and clamps auto-labeled SAE features during ESM-2 generation with dose-response, mean-activation-addition baseline, and preserved-plausibility gates.
**Final Verdict**: READY
**Date**: 2026-07-15

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (given-behavior variant): `idea-stage/IDEA_REPORT.md`
- Landscape (grounding only — no external URLs retrieved this run due to project policy blocking arxiv-cutoff 2412+ and the reference paper by title): `idea-stage/LANDSCAPE.md`
- Retrieval audit trail: `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot
- **Dominant contribution**: protocol-pinned faithful reproduction with matched-capacity specificity controls at every claim.
- **Optional supporting contribution**: sensitivity sweep on τ_F1 × q_top (M2 appendix) transparently reports headline-number robustness.
- **Explicitly rejected complexity**: SAE retraining; ESM-2 tuning; pretraining-formation tracing; scale-generalization swaps (deferred to `/auto-verify`); novelty search (skipped for `BEHAVIOR_SOURCE=given`).

## Must-Prove Claims
- c1 — SAE per-layer interpretable count ~2,548 vs. neurons ≤ ~254 at same layer (≥10× ratio).
- c2 — SAE covers ~143 Swiss-Prot concepts vs. neurons ~46 covered / ~15 clean.
- c3 — c2 gap survives PCA / random-rotation / shuffled-SAE controls, licensing the superposition interpretation.
- c4 — ≥10% of Swiss-Prot-unaligned SAE features receive coherent, non-synonym LLM labels above random-feature control.
- c5a — SAE-linear-probe > neuron-linear-probe on held-out Swiss-Prot annotation-filling (paired-Wilcoxon p < 0.05, K=50 concepts).
- c5b — SAE-feature-clamp steering shows monotone dose-response yield above no-steer / random-clamp / mean-activation-addition, within plausibility band, on ≥1 concrete target property.

## First Runs to Launch
1. `m1_L1` through `m1_L33` — six parallel launches of the per-layer feature-count harness (uses all four GPUs; small per-layer footprint).
2. `m2` — concept-alignment harness (depends on M1 activations cache).
3. `m3_seed{42,43,44}` in parallel with `m4` and `m5` (all depend only on M2, so they fan out).

## Main Risks
- **Metric-choice sensitivity** (Gap G1): headline numbers ~2,548 / ~143 / ~46 depend on τ_F1 and q_top. Mitigation: sensitivity sweep in M2 appendix; report the sweep transparently, not just the primary point.
- **LLM endpoint reliability** (Gap G2): `gpt-5.4` at `dmxapi.cn` may rate-limit or return malformed responses. Mitigation: rigid JSON schema, retry once on parse failure, cache raw responses, fall back to smaller `n_features` in M4 if the endpoint is slow.
- **Steering off-target drift** (Gap G3): clamping may boost yield while destroying plausibility. Mitigation: pseudo-perplexity plausibility band in M6, reported alongside yield.
- **GPU budget** (10h): M6 is the largest single milestone at 3.0h; if M1–M5 overrun their allocations the runner downgrades M6's `M_features` from 4 to 2 rather than skipping the milestone.

## Next Action
- `/mechanism-skills` — route Unit Interpretation → Decision Auditing → Causal Intervention to concrete method families + submethods (auto-interp submethod; steering submethod: feature-clamp vs. activation-addition).
- Then `/auto-experiment` — implement + deploy from the routing + plan.
