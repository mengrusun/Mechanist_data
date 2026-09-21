# Verify Plan — C2: dose-response helix

## Claim
Amplifying the α-helix feature set (σ_proj-unit dosing) during autoregressive DNA generation increases the encoded protein's pLDDT-weighted α-helix fraction, monotonically up to a mapped interior optimum, reproduced across two structure predictors.

**Main-experiment verdict:** supported
**Main-experiment metric:** primary endpoint helix_hgi_w (pLDDT-weighted HGI helix fraction); ESMFold ρ=0.867 p=0.0012 Δ+0.129; OmegaFold ρ=0.879 p=0.0008 Δ+0.135; interior c*=21.48

## Main experiment (from /auto-experiment)
- Method: σ_proj-unit dose-response sweep, Spearman ρ on held-out seeds 200,201
- Dataset: natural-CDS Evo2-7B generation at c* and all dose grid points
- Model (readout): pLDDT-weighted HGI helix fraction (helix_hgi_w) from ESMFold + OmegaFold
- Metric: Spearman ρ on (c_sigma, helix_hgi_w) over 10 doses, held-out seeds 200+201

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | H-only pLDDT-weighted helix fraction (helix_h_w) as primary endpoint | HGI pLDDT-weighted helix fraction (helix_hgi_w) | Tests if the dose-response trend is robust to the helix-assignment model. H-only excludes 3-10 G and pi I helices; if the trend depends on these rarer helix types being assigned to S's features, H-only would show weaker trend. M2 result files store both helix_hgi_w_mean and helix_h_w_mean per run — no re-run needed. | EXPERIMENT_PLAN.md §M0 helix_def list; M2 result files store helix_h_w_mean |

## Success Criterion (per variant)
Claim is supported if Spearman ρ on H-only endpoint is positive and significant (p<0.05) for both predictors, with Δ(c*) vs c=0 > 0 on the held-out seeds. The trend direction must match the main experiment (positive).

## Reviewer Notes (Phase 4)
The H-only swap is the most direct test of the helix-definition dependency for C2. The M2 files store both helix metrics. Using H-only is more conservative (G and I helices account for ~5-15% of helix content in typical proteins). If ρ remains ~0.87-0.88 on H-only, the dose-response is robust to this labeling choice. This is a genuine test because H-only is the strict definition used in many structure papers.

Implementation: read all M2 result files, extract helix_h_w_mean per dose per predictor, compute Spearman ρ on held-out seeds. No GPU needed.
