## C3 — Helix-axis specificity: S vs 48-direction random null + matched-control  (β-arm = documented negative)

| Predictor | S Δ helix_hgi_w [95% CI] | Matched-control Δ | Null mean | Null max | n_null ≥ S | Empirical p (1-sided) | z-score |
|---|---|---|---|---|---|---|---|
| ESMFold   | **+0.1291** [0.0996, 0.1585] (283 clusters) | -0.0312 | -0.0041 | +0.0564 | 0/48 | **0.0204** | **5.30** |
| OmegaFold | **+0.1350** [0.1092, 0.1610] (283 clusters) | -0.0245 | -0.0010 | +0.0590 | 0/48 | **0.0204** | **5.80** |

**Verify H-only swap:** ESMFold z = 5.83, OmegaFold z = 6.13 (both 0/48 null ≥ S) — specificity strengthens under the DSSP helix-definition swap.

**Capability match (ESMFold):** S valid-ORF = 0.895, null-mean valid-ORF = 0.882, c0 valid-ORF = 0.902  →  arms match on capability.  **Not a pLDDT-confidence artifact:** S mean pLDDT (59.7 ESMFold / 66.3 OmegaFold) is _lower_ than baseline c0 (62.8 / 68.7).  **β-arm:** β_v2 (features 22326, 21653, 13992, 17067, 31467; set-AUROC 0.826 on β labels — same M0 bar as helix) did NOT clear the sheet-raising bar at c* → C3 is helix-axis specificity, NOT symmetric double dissociation (documented negative per plan P9).

_Source: `results/m3_specificity_summary.json` + 16 chunk files `results/m3_random_c21.4801_d*.json`._
_Caveat: with n_null = 48, empirical one-sided p is floored at ~1/(48+1) ≈ 0.0204; the z-score (which uses null mean/variance) is the stronger statistic._
