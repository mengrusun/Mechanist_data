# Pipeline Summary

**Problem**: Steer Evo2-7B to generate DNA with higher α-helical protein content by amplifying Layer-26 Mixed SAE features (reproduction; behavior & mechanism given).
**Final Method Thesis**: Identify α-helix-selective SAE latents on labeled coding DNA, amplify exactly those during autoregressive decoding, and measure the resulting increase in predicted %-helix of translated ORFs, sweeping α to find the optimum under a fixed validity constraint.
**Final Verdict**: READY
**Date**: 2026-08-19
**resource_fidelity**: strict | **chosen_mechanism**: SAE feature amplification (Evo-2 Layer-26 Mixed) | **mechanism_strategy**: n/a | **M0**: none (behavior=given)

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Captured claims: `idea-stage/IDEA_REPORT.md` | Landscape: `idea-stage/LANDSCAPE.md`

## Contribution Snapshot
- Dominant: dose- & quality-controlled causal α-helix steering of a DNA LM via its released SAE, with reported optimal α*.
- Supporting: leakage-free SAE-feature-selectivity confirmation for a structural phenotype on coding DNA.
- Rejected complexity: SAE retraining, fine-tuning/weight-editing, optimized steering vectors, any model/data downscaling.

## Must-Prove Claims
- C1: α-helix-selective SAE feature set exists (held-out, above null & β/coil controls).
- C2: amplifying it raises predicted %-helix vs matched unsteered baseline (QC subset, held-out).
- C3: dose-response with an identifiable optimal α* (validation-selected, held-out-confirmed).

## First Runs to Launch
1. S0 — build `data/ecoli_labeled.parquet` (RefSeq CDS + AlphaFold-DB + DSSP labels, homology-clustered splits).
2. M1 — capture Evo2 layer-26 SAE activations, select & validate α-helix feature set `S` (C1).
3. M2 — baseline + α-grid dev generation, ESMFold→DSSP %-helix, freeze α* (C2/C3-dev).

## Main Risks
- Degenerate optimum (validity destroyed): mitigated by treatment-independent QC + ITT + validity floor.
- Predicted readout / ESMFold offline: dual-confidence reporting + local SS-predictor fallback.
- Folding throughput vs 8 h budget: trim generation N to floors, never model/SAE scale.

## Next Action
- Proceed to `/auto-experiment` (mechanism family already committed: CHOSEN_FAMILY = SAE amplification; no /mechanism-skills routing needed).
