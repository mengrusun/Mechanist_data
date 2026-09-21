# Verification Report

**Date**: 2026-07-20
**Swap variants**: true — full 3-stage pipeline (Stages 1–3)
**Dimensions tested**: model (helix-labeling model / structural readout model swap)
**Variants/claim**: 1 (one swap per listed axis)
**Threshold**: robustness >= 0.50, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: PASS (all 3 claims); see per-claim sub-audit breakdown below.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|-----------------------------------------------|---------------------------------------|-------------------------------|------------|-------|-------|
| C1 | helix feature set exists | supported | PASS (exp PASS / mech N/A) | clean | 1/1 | 1.00 | PASS | H-only AUROC 0.906/0.868 ≈ HGI 0.901/0.866 — helix-def-model robust |
| C2 | dose-response helix | supported | PASS (exp PASS / mech PASS) | clean | 1/1 | 1.00 | PASS | H-only rho=0.867/0.903 p<0.001 both; delta slightly larger under H-only |
| C3 | specificity causal knob | supported | PASS (exp PASS / mech PASS) | clean | 1/1 | 1.00 | PASS | 0/48 null >= S, z=5.83/6.13 under H-only — specificity strengthens |

> **Column glossary**:
> - **Main-experiment verdict** — main experiment's own conclusion from `refine-logs/main-experiment-verdicts.json`.
> - **Main-experiment integrity (Phase 2, combined)** — `max_severity(experiment-audit, mechanism-audit)` on `refine-logs/` scoped to each claim.
> - **Variant integrity (Phase 9, combined)** — same rule on per-claim variant directory.
> - **Eligible variants** — variants with `integrity_status ∈ {pass, warn}`, contributing to robustness.
> - **Robustness** — `#pass / N_eligible` where `pass` = `consistent_with_main_experiment = pass`.

## Swap Details

### C1 model swap
- **Replaced**: DSSP HGI helix label model (H+G+I helix types)
- **With**: DSSP H-only strict alpha helix model (excludes 3-10 G and pi I)
- **Data source**: pre-computed M0 H_only result files (no GPU re-run; M0 ran all 12 configs)
- **Key result**: prokaryote set-AUROC 0.906 (±0.002, n=3 seeds); eukaryote 0.868 (±0.004); shuffle-null ~0.48; both above ≥0.80 threshold

### C2 model swap
- **Replaced**: pLDDT-weighted HGI helix fraction (helix_hgi_w) as primary endpoint
- **With**: pLDDT-weighted H-only helix fraction (helix_h_w)
- **Data source**: pre-computed M2 result files (helix_h_w_mean stored per run; no GPU re-run)
- **Key result**: ESMFold Spearman rho=0.867 p=0.0012, Δ+0.144; OmegaFold rho=0.903 p=0.0003, Δ+0.147; both significant, dual-predictor agree

### C3 model swap
- **Replaced**: pLDDT-weighted HGI specificity endpoint (helix_hgi_w)
- **With**: pLDDT-weighted H-only specificity endpoint (helix_h_w)
- **Data source**: pre-computed M3 arm + null chunk files (helix_h_w_mean stored per arm/direction; no GPU re-run)
- **Key result**: 0/48 null dirs >= S under both predictors (ESMFold z=5.83 p=0.020; OmegaFold z=6.13 p=0.020); matched-control Δ negative

## Integrity Audit

**Overall**: PASS — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) + Phase 9 (variants) findings.

- Phase 2 (main experiment): all 3 claims PASS (C1: exp PASS / mech N/A; C2: exp PASS / mech PASS; C3: exp PASS / mech PASS)
- Phase 9 (variants): all 3 variants PASS (experiment and mechanism audits clean for all)
- Minor noted caveat for C3 variant: per-sample H-only values not stored in M3 files → cluster-bootstrap CI not available for H-only variant; empirical p-value and z-score test remain valid. Does not affect integrity verdict.

## Stage-2 Selection

All 3 admitted claims were picked (cap = MAX_VERIFY_CLAIMS = 3 = |ADMITTED|; no deferral needed).

## Details
- C1: `verify/C1_helix_feature_set_exists/ROBUSTNESS.md`
- C2: `verify/C2_dose_response_helix/ROBUSTNESS.md`
- C3: `verify/C3_specificity_causal_knob/ROBUSTNESS.md`

## GPU-hours spent
~0 GPU-hours (all variants are analysis-only — re-analysis of pre-computed M0/M2/M3 result files with H-only metric extraction; no new generation or structure folding).

## Next Step

→ **All claims PASS** → proceed to `/auto-iteration-loop "hardening the alpha-helix feature-steering knob in Evo2-7B (Round 2)"`. The robustness story is positive: C1 (existence), C2 (dose-response), and C3 (specificity) all hold under the H-only helix-definition model swap — the frozen S features are robust to the choice of DSSP helix type enumeration. The paper narrative should note this cross-definition robustness explicitly as an additional strength.
