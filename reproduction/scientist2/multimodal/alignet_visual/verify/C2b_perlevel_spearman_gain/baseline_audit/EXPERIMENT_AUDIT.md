# Experiment Audit — C2b: Per-Level Spearman Gain Holds at Coarse/Mid/Fine

**Claim**: The aggregate Spearman gain in C2a holds separately at each of the coarse / mid / fine abstraction levels (per-level Δρ > 0 for all three levels).

**Scope**: M3+M4 per-level output in `runs/M3_aligned_dinov2/eval_things_multilevel.json` and `runs/M4_unaligned_dinov2/eval_things_multilevel.json`

---

## Check A — Ground-Truth Provenance

**Finding**: PASS. Per-level human similarity is derived from THINGS training triplets, stratified by level buckets from THINGS categorical metadata (`Top-down Category (WordNet)` column). Real human behavioral data. Same as C2a — PASS.

**Severity**: PASS

---

## Check B — Score Normalization

**Finding**: PASS. Same as C2a — Spearman on raw cosine similarity vs human co-selection fractions; no model-specific normalization.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS with caveat. Per-level numbers in the on-disk JSONs:
- M3: coarse Spearman=0.707 (183 pairs), mid=0.611 (1,194 pairs), fine=NaN (0 pairs)
- M4: coarse Spearman=0.182 (13,861 pairs), mid=0.184 (328,455 pairs), fine=NaN (0 pairs)

EXPERIMENT_RESULTS.md cites:
- coarse Δρ=+0.526, mid Δρ=+0.427 — both match the on-disk difference (0.707−0.181=0.526; 0.611−0.184=0.427) ✓
- fine: cited as "nan" / "0 pairs" — matches on-disk (n_pairs_fine=0) ✓

One minor discrepancy: M4's spearman_coarse in the JSON is 0.182 (n_pairs=13,861) but the RESULTS summary says 0.181. This rounds identically.

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. Same code path as C2a; per-level computation is exercised in `eval_features()` via `level_bucket_triplets()`.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: WARN. The claim requires Δρ > 0 "at each of the coarse / mid / fine levels." But the fine level has 0 usable Spearman pairs from the THINGS categorical metadata stratification — the per-level claim at fine is unverifiable from the main experiment data. The CLAIMS_LEDGER and EXPERIMENT_RESULTS.md correctly mark C2b as "conditional" for this reason. However, the claim statement itself does not carry this qualifier — it says "all three levels." This is a scope over-claim relative to what was actually tested.

**Severity**: WARN

---

## Check F — Evaluation Type

**Finding**: real_gt — same as C2a.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: warn**

C2b is admitted with a WARN: the fine-level Spearman is uncomputable (0 pairs) from the current categorical stratification, making the "at each of coarse/mid/fine" predicate partially unverifiable. C2b is admitted to the ADMITTED pool for Stage 2, but will be INTEGRITY_ONLY due to the MAX_VERIFY_CLAIMS=1 cap after C2a is picked.
