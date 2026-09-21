## C2: robustness = 1.00 (threshold = 0.5, eligible = 2/2)  →  ✅ PASS  `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]`

- verdict: PASS
- swap_variants_run: true
- Main-experiment verdict on C2: supported (Pearson r=0.987, n=200)
- Main-experiment integrity (Phase 2): WARN (warn_source=experiment — scope/verification note; admitted)
- Variant integrity (Phase 9): WARN (scope-only; 0 fail, both eligible)
- Variant counts (over `consistent_with_main_experiment`): 2 pass, 0 fail  (of 2 eligible; 0 excluded for integrity)
- **Method dimension** (GOR-style windowed logistic, no ESM2, Q3=0.658): matches the main experiment — consistent=pass, claim_supported=pass. Pearson r(gor %H, expDSSP %H) = **0.8505** (n=200, p=3.8e-57), frame recovery 1.000. r drops from 0.987→0.851 but stays far above the 0.7 pass bar, same positive direction. `[INTEGRITY: WARN — experiment]`
- **Dataset dimension** (same ESM2+probe metric on 310 fresh unseen deduplicated proteins): matches the main experiment — consistent=pass, claim_supported=pass. Pearson r(fast %H, expDSSP %H) = **0.9792** (n=310, p=1.9e-215), Q3=0.859, frame recovery 1.000. Essentially unchanged from 0.987. `[INTEGRITY: WARN — experiment]`

Interpretation: C2's conclusion — a fast sequence-based α-helical-content metric agrees with structure-based experimental-DSSP %H at r ≥ 0.7 — is **robust under both swaps**. The method swap is the more informative axis: a fully independent classical predictor with far weaker per-residue accuracy (GOR Q3 0.658 vs ESM2 0.858) still tracks DSSP %H at r=0.85, showing the agreement is a general property of protein-level %H estimation rather than an ESM2-specific artifact. The dataset swap shows the r=0.987 correlation reproduces (r=0.979) on unseen proteins, ruling out an eval-set-selection artifact. Both robustness axes reach the pass bar; no variant failed integrity. The only caveat is scope (one run per variant on finite sets), carried as a WARN — the numbers stand for the tested conditions. No iteration back-edge needed for C2.
