## C3: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (no Bonferroni correction for 3 language tests; FR sign reversal breaks 'all 3 langs' predicate; no on-disk record of manual translation spot-check)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C5 selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C3's not-supported verdict reflects the intrinsically weak honesty signal (EN shift=+0.14 within judge std=1.25, tracking through cross-lingual evaluation as noise). The Wilcoxon paired test is the correct choice; the absence of Bonferroni correction is noted but not misleading (all p > 0.10 far above even uncorrected threshold). FR sign reversal (shift=-0.10) is a genuine finding. To further investigate: `/auto-verify C3 — resume: true` with a model swap to test if a different 8B model shows stronger cross-lingual honesty transfer, or a dataset swap using a stronger/more discriminative honesty concept.
