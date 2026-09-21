# Experiment Audit — C3 (specificity causal knob)
**Scope:** M3 (refine-logs/EXPERIMENT_PLAN.md#M3); result files results/m3_specificity_summary.json, results/m3_*.json

**Claim:** At the capability-preserved interior dose, the α-helix rise is specific to S — S's helix gain exceeds a ≥30-direction norm-matched random-null (and matched-control) with a tight effect-size CI, reproduced across two predictors — establishing S as a causally manipulable, specific knob (helix-axis specificity; β-arm resolved).

## Check A — GT Provenance
**Result: PASS (same as C2)**
Helix fraction from structure prediction (ESMFold + OmegaFold) on generated sequences. Appropriate and by design. The null directions generate sequences via the same pipeline, so the comparison is fair — same prediction pipeline for S and all null directions.

## Check B — Score Normalization
**Result: PASS**
The S-vs-null specificity statistic computes S's Δhelix_hgi_w vs each null direction's Δhelix_hgi_w (both measured the same way). No model-max normalization. Empirical p-value is rank-based (fraction of null dirs ≥ S).

## Check C — Result File Existence (claim-scoped)
**Result: PASS**
- results/m3_specificity_summary.json: exists, contains PRIMARY_random_null per predictor with n_directions=48, n_null_ge_S=0, empirical_one_sided_p=0.020, z_score=5.296 (ESMFold) / 5.803 (OmegaFold).
- M3 arm files: all 36 per-arm files (S, matched-control, beta × 4 c-values × 3 seeds) verified present.
- Random null files: 16 JSON files covering dir ranges 0-47 (48 directions in 3-direction chunks) verified present.
- Cited values (delta +0.129/+0.135, CI [0.100,0.159]/[0.109,0.161], z=5.30/5.80, p=0.020, 0/48 null ≥ S) all confirmed in m3_specificity_summary.json.

## Check D — Dead Code
**Result: PASS**
m3_specificity.py, m3_random2.py (with checkpoint+resume), m3_consolidate_v2.py all produced output files. Checkpoint+resume pattern active for null runs (noted in CLAIMS_LEDGER.md — timeout bug repaired by adding per-phase checkpointing; zero scientific loss confirmed).

## Check E — Scope (claim-scoped)
**Result: PASS**
Claim says "≥30-direction norm-matched random-null" — experiment ran 48 directions (>30, exceeds requirement). Claim says "reproduced across two predictors" — both ESMFold and OmegaFold show 0/48 null dirs ≥ S. Matched-control also included. β-arm resolved with documented-negative verdict (plan P9; clears the bar for helix-axis specificity framing). No over-claim.

## Check F — Evaluation Type
**Result: synthetic_proxy**
Same as C2 — structure prediction on generated sequences; appropriate and by design.

## Overall Verdict
**overall_verdict: pass**

C3 specificity test is methodologically sound: 48 norm-matched null directions (exceeds ≥30 requirement), dual-predictor confirmation, matched-control negative, capability matched at c* (S valid-ORF 0.895 ≈ null 0.882 ≈ c0 0.902). All result files present. The helix-axis specificity framing is correctly grounded in plan P9 (β-arm honest negative does not weaken the S-vs-null conclusion).
