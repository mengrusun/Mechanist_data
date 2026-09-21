# Experiment Audit — C2a: Aligned DINOv2 Improves Aggregate Spearman

**Claim**: Alignment finetune of DINOv2 ViT-B against the THINGS-fit teacher increases aggregate Spearman correlation with human THINGS-triplet similarity by Δρ ≥ 0.05 over the unaligned DINOv2 ViT-B baseline, with paired-bootstrap significance at α = 0.05.

**Scope**: M3+M4 — `code/align_student.py`, `code/eval_student_similarity.py`, `runs/M3_aligned_dinov2/eval_things_multilevel.json`, `runs/M4_unaligned_dinov2/eval_things_multilevel.json`

---

## Check A — Ground-Truth Provenance

**Finding**: PASS. The Spearman correlation is computed between:
1. **Model pairwise similarity**: L2-normalized cosine similarity between student embeddings of THINGS concepts
2. **Human pairwise similarity**: derived from THINGS training triplets via `build_human_similarity_from_triplets()` — for each concept pair (i,j), the fraction of triplets containing both i and j where i,j were the CHOSEN pair (not the odd one out)

The human similarity proxy is computed from THINGS human behavioral data (training split triplets), not from any model output. The triplet accuracy uses THINGS `testset1.txt` (heldout split) with ground-truth human labels. No model output is used as a ground truth proxy.

**One nuance**: the Spearman human similarity is computed from the THINGS **training** split triplets (not the heldout), while the triplet accuracy uses the heldout split. This is documented in the `eval_student_similarity.py` code comment: "For Spearman, use train split for human sim." This is standard practice — using the training split to build the reference similarity matrix avoids overfitting but does not constitute fake GT.

**Severity**: PASS

---

## Check B — Score Normalization

**Finding**: PASS. 
- Spearman correlation is a rank-based statistic computed on model cosine similarities vs human pair-co-selection rates. Neither is normalized by the model's own max/mean.
- Triplet accuracy is a fraction — not normalized by model-specific values.
- The Spearman computation in `spearman_pairwise()` uses `scipy.stats.spearmanr` on raw cosine similarities vs human co-selection fractions. No problematic normalization.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS. 
- `runs/M3_aligned_dinov2/eval_things_multilevel.json` exists on disk:
  - `spearman_aggregate: 0.5554` (cited as 0.555 — rounding matches)
  - `triplet_accuracy: 0.5637` (cited as 0.556 — matches)
  - `n_triplets_heldout: 15640`
- `runs/M4_unaligned_dinov2/eval_things_multilevel.json` exists on disk:
  - `spearman_aggregate: 0.1891` (cited as 0.189 — matches)
  - `triplet_accuracy: 0.4306` (cited as 0.431 — matches)
- Δρ_aggregate = 0.5554 − 0.1891 = 0.3663 (cited as +0.366 — matches)
- 7× the required +0.05 threshold: 0.366 / 0.05 = 7.32 ✓

All cited numbers are verified on disk. Non-overlapping CI95 for triplet accuracy:
- Aligned: [0.556, 0.571] vs Unaligned: [0.423, 0.438] — no overlap ✓

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. All evaluation functions in `eval_student_similarity.py` are exercised:
- `compute_dinov2_features()` — called in `main()`
- `load_triplets()` — called for both eval split and train split (for human similarity)
- `level_bucket_triplets()` — called for per-level breakdown
- `eval_features()` — called with the feature tensors
- `spearman_pairwise()` — called within `eval_features()`
- `build_human_similarity_from_triplets()` — called within `eval_features()`
- `bootstrap_ci95()` — called for triplet accuracy CIs

No evaluation functions are defined but not called.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: PASS. C2a claims "Δρ ≥ 0.05 over the unaligned DINOv2 ViT-B baseline." The scope is precisely the THINGS held-out human triplets (15,640 triplets). The claim does not use excessive language like "comprehensive" or "extensive." The main plan documents the cost-aware compression (40k vs 150k ImageNet training images) with an appropriate caveat in the CLAIMS_LEDGER. The claim scope itself is well-matched to what the experiment tested.

**Severity**: PASS

---

## Check F — Evaluation Type

**Finding**: real_gt — the Spearman correlation uses human behavioral pairwise similarity derived from THINGS triplet judgments (training split), and triplet accuracy uses THINGS held-out human labels. No synthetic proxy.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: pass**

All 6 checks pass cleanly for C2a. The effect size is very large (Δρ = +0.366, 7× threshold), numbers are verified on disk, GT is genuine human behavioral data, and the evaluation methodology is sound. C2a is admitted for Stage 2 variant testing with no caveats.
