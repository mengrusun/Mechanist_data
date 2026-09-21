# Verify Plan — C3: specificity causal knob

## Claim
At the capability-preserved interior dose, the α-helix rise is specific to S — S's helix gain exceeds a ≥30-direction norm-matched random-null (and matched-control) with a tight effect-size CI, reproduced across two predictors — establishing S as a causally manipulable, specific knob (helix-axis specificity; β-arm resolved).

**Main-experiment verdict:** supported
**Main-experiment metric:** 0/48 null dirs ≥ S; ESMFold z=5.30 p=0.020; OmegaFold z=5.80 p=0.020; S Δhelix_hgi_w +0.129/+0.135, CI excludes 0

## Main experiment (from /auto-experiment)
- Method: S-vs-48-direction norm-matched random null at locked interior c*, pLDDT-weighted HGI helix fraction (helix_hgi_w) as primary endpoint
- Dataset: Evo2-7B generation at c*=21.48 for S arm, 48 random null directions, matched control
- Model (readout): pLDDT-weighted HGI helix fraction from ESMFold + OmegaFold
- Metric: empirical one-sided p = (1+n_null_ge_S)/(N+1), z-score vs null distribution

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | H-only pLDDT-weighted helix fraction (helix_h_w) as specificity endpoint | HGI pLDDT-weighted helix fraction (helix_hgi_w) | Tests if S's advantage over 48 norm-matched null directions holds when the helix labeling model uses strict alpha-only definition. Directly probes whether the specificity signal rests on G/I-helix contributions. M3 arm + null files store helix_h_w_mean per predictor — no re-run needed. | EXPERIMENT_PLAN.md §M0 helix_def; M3 result files store helix_h_w_mean |

## Success Criterion (per variant)
Claim is supported if 0/N null directions (or < 5% = ≤2 out of 48) match or exceed S's H-only Δhelix_h_w under BOTH predictors, with z-score ≥ 2.0 (conservative) per predictor.

## Reviewer Notes (Phase 4)
The H-only swap is the cleanest test for C3: if the 48-direction advantage (0/48) holds under H-only, it demonstrates the specificity is not dependent on 3-10 or pi helix types. The matched-control result should also be checked. M3 arm files and null chunk files both store aggregate helix_h_w_mean per predictor per direction/arm.

Implementation: read M3 arm files for S at c* (seeds 200,201), matched-control at c*, and c=0 baseline; read all 16 null chunk JSON files; extract helix_h_w_mean per predictor. Compute S_delta_h = S_helix_h_w - baseline_helix_h_w and analogously for nulls. No GPU needed.
