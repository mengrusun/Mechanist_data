## C2 — σ_proj-unit dose-response of pLDDT-weighted α-helix fraction (held-out seeds 200, 201)

| c (σ_proj) | ESMFold mean [95% CI] | OmegaFold mean [95% CI] | valid-ORF | region |
|---|---|---|---|---|
| -5.3700 | 0.463 [0.440, 0.489] | 0.479 [0.455, 0.505] | 0.885 | below baseline |
| 0.0000 | 0.454 [0.429, 0.478] | 0.473 [0.450, 0.496] | 0.902 | capability-preserved |
| 2.6850 | 0.450 [0.426, 0.476] | 0.467 [0.444, 0.491] | 0.897 | capability-preserved |
| 5.3700 | 0.451 [0.427, 0.477] | 0.469 [0.445, 0.496] | 0.908 | capability-preserved |
| 10.7400 | 0.479 [0.454, 0.505] | 0.500 [0.475, 0.525] | 0.902 | capability-preserved |
| 21.4801 **★ c*** | 0.593 [0.566, 0.621] | 0.612 [0.586, 0.639] | 0.895 | capability-preserved |
| 32.2201 | 0.543 [0.517, 0.568] | 0.617 [0.592, 0.641] | 0.807 | degraded (excluded) |
| 42.9601 | 0.732 [0.698, 0.764] | 0.766 [0.739, 0.790] | 0.648 | degraded (excluded) |
| 64.4402 | 0.885 [0.868, 0.899] | 0.889 [0.877, 0.900] | 0.770 | degraded (excluded) |
| 85.9203 | 0.858 [0.840, 0.876] | 0.870 [0.858, 0.882] | 0.820 | degraded (excluded) |

**Primary trend (held-out seeds, pLDDT-weighted endpoint):**
- ESMFold:   Spearman ρ = **0.867**, p = **0.0012**;   Δ(c*) = **+0.129** [95% cluster-bootstrap CI 0.100, 0.159]  (283 clusters)
- OmegaFold: Spearman ρ = **0.879**, p = **0.0008**;   Δ(c*) = **+0.135** [95% CI 0.109, 0.161]

_Source: `results/m2_dose_response_curve.json`._ c* = pLDDT-weighted-helix argmax within the contiguous capability-preserved region (valid-ORF ≥ 0.95×baseline; region ends at c=32.22 where valid-ORF collapses to 0.807). High-dose helix resurgences at c ∈ {43, 64, 86} are in the degraded regime and correctly excluded (plan P7).
