# Phase 6 cross-model code review (gpt-5.6-luna) — resolution

No CRITICAL blockers. MAJOR issues fixed before deploy:
- **Bootstrap p-value → permutation test**: both variants now use a two-sample PERMUTATION test (label shuffle, two-sided, +1 correction) for the primary p-value, with the bootstrap only for the CI. (V1 analyze_diffmeans.diff_stats; V2 seqonly_readout.perm_p/diff_stats.)
- **V2 test-set leakage via early stopping**: predictor now FITs on train, EARLY-STOPS on the existing held-out VAL split, and reports FINAL metrics on the untouched TEST split.
- **V2 filtering on old readout**: cached generations filtered ONLY on pre-readout criteria (qc_ok & protein); the new predictor runs on all such sequences; ESM-2 reference uses its own finite-value eligibility. Per-filter sample counts recorded.
- **V1 α*-selection**: added N_min=150 QC-valid floor, deterministic smallest-α tie-break, NaN guard, and α*=0 fallback (no positive α passing the floor is itself an informative null).
- **V1 frame alignment**: fail-closed length assertion on captured tokens vs 3L (same codon=mean-over-3nt convention as validated m1_capture).

Retained-by-design (documented): unpaired two-sample conditional test MATCHES the main experiment's stated procedure (seed pairing gives ~no variance reduction once steering diverges the generated sequences). V1 baseline freshly generated in the identical code path; V1 α self-calibrated on dev (not transplanted); V2 generations held byte-identical. DSSP labels are real GT for both.
