# Review Summary — Testing-Method Refinement

**Date**: 2026-08-19 | Reviewer: external LLM (llm-chat), senior ML + comp-bio methods referee | Focus: verification methodology only (claims/mechanism fixed).

## What was reviewed
The testing method for the 3 fixed claims (C1 feature selectivity, C2 steered-vs-baseline helix gain, C3 optimal α*). The reviewer was told the behavior, mechanism, model, and SAE are fixed and must not be renegotiated.

## Top methodological risks raised → how the plan resolves them

1. **Degenerate optimum** (helix metric won by destroying valid coding sequences). → Treatment-independent QC frozen before generation; identical QC for baseline & steered; ITT + conditional reporting; α* validity floor (ORF-valid rate + perplexity); composition-adjusted effect. (FINAL_PROPOSAL Prereg-5,7,8; PLAN M2/M3 predicates + budget guard.)
2. **Feature-selection leakage** (homologs across splits; nucleotide pseudo-replication; unmatched controls). → mmseqs2 homology-cluster split; gene/cluster as statistical unit; codon-mean activation; gene+codon-position-preserving shuffle null; BH-FDR over 32768; β/coil controls matched. (PLAN S0, M1.)
3. **Structure-label leakage / bad DNA→residue mapping**. → exact CDS→protein→structure alignment, drop mismatches; predefined codon→residue label rule; AlphaFold-DB labels kept independent of the ESMFold readout. (PLAN S0.)
4. **Under-specified steering for a TopK SAE**. → single frozen primary intervention (residual add of unit decoder directions scaled by median-positive activation; α portable), secondary pre-TopK variant; sanity telemetry (target activation, TopK support overlap, residual-norm/KL). (FINAL_PROPOSAL Prereg-1; PLAN M2 telemetry.)
5. **Baseline unfairness**. → identical decoding, paired matched random-seed streams, α=0 through the same code path verified numerically equal. (FINAL_PROPOSAL Prereg-2; PLAN M2.)
6. **α* winner's-curse / grid overfit**. → α grid+seeds+floor preregistered; α* chosen on validation/dev, confirmed on **disjoint** held-out seed block H; simultaneous CI band; replication. (FINAL_PROPOSAL Prereg-8; PLAN M2→M3 split.)
7. **Power / effective sample size**. → cluster-level power, ≥500 QC-valid proteins/arm in confirmation, gene/sequence-cluster bootstrap, FDR correction. (PLAN M1/M3.)
8. **Predicted (not experimental) readout**. → dual all-residue vs pLDDT≥70 reporting; "predicted helix" phrasing; random/β-sheet steering controls; optional second predictor. (FINAL_PROPOSAL Prereg-6, risks; PLAN M-CTRL.)
9. **Weak off-target controls**. → random-feature, β-sheet-feature, null-direction arms; composition/length/entropy/GC deltas with CIs. (PLAN M-CTRL.)

## Non-falsifiability items explicitly eliminated
All of: numeric definition of "small subset"/"higher"/"quality-controlled"/"optimum"; splits & thresholds frozen before test; no treatment-dependent QC; α* selected and evaluated on disjoint data; leakage-free selection; exact TopK intervention; fixed failure-handling; full reporting (no best-seed cherry-pick).

## Verdict
Testing method **READY** for the reproduction combination. Claims unchanged (as required). Remaining external dependency: ESMFold weight availability offline (mitigation: local SS-predictor fallback as an eval-tool choice; does not affect strict model/SAE fidelity).
