# Experiment Audit — C4a: Aligned Non-Inferior on Downstream One-Shot

**Claim**: The aligned DINOv2 ViT-B is non-inferior to the unaligned baseline on downstream one-shot classification: mean top-1 across the 4-dataset stratified subset (Birds / UC Merced / Colon-pathology / Aircraft) is ≥ the unaligned mean, with no per-dataset drop > 1 point uncompensated elsewhere.

**Scope**: M7 — `code/eval_downstream.py`, `runs/M7_downstream/results.json`

---

## Check A — Ground-Truth Provenance

**Finding**: WARN. The claim specifies Birds / UC Merced / Colon-pathology / Aircraft as the evaluation panel. The main experiment used dataset substitutes: DTD (47-class textures), Fashion-MNIST (10-class clothing), imagenet_val_top100, imagenet_val_top20. The on-disk datasets have their own true ground-truth labels (DTD label set, Fashion-MNIST label set, etc.) — so GT provenance for the substitute datasets is real.

However, the **dataset substitution itself** introduces a scope issue: the claim states a specific panel (Birds/UC-Merced/Colon/Aircraft) that was never tested. The ground truth is real for the substitute panel, but the claim scope is not matched by the evaluation scope. This is a documented limitation (open item in CLAIMS_LEDGER), not hidden.

**Severity**: WARN — scope mismatch between claimed panel and tested panel. Not FAIL because the substitution is honestly documented throughout.

---

## Check B — Score Normalization

**Finding**: PASS. Top-1 accuracy is the fraction of test items correctly classified — not normalized by model max. Standard 1-shot evaluation metric.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS. `runs/M7_downstream/results.json` exists on disk:
- dtd: aligned=0.247, unaligned=0.497, Δ=−0.250 (cited as −0.250 ✓)
- fashion_mnist: aligned=0.382, unaligned=0.584, Δ=−0.202 (cited as −0.202 ✓)
- imagenet_val_top100: aligned=0.370, unaligned=0.740, Δ=−0.370 (cited as −0.369 ✓, rounding)
- imagenet_val_top20: aligned=0.622, unaligned=0.936, Δ=−0.314 (cited as −0.314 ✓)
- mean_top1_aligned: 0.405 (cited as 0.405 ✓), mean_top1_unaligned: 0.689 (cited as 0.689 ✓)
- All p_positive=0.00 confirmed

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. `eval_downstream.py` runs the 1-shot evaluation for each (student, dataset) pair; the paired-bootstrap is computed and saved. No dead evaluation paths.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: WARN. The claim says "4-dataset stratified subset (Birds / UC Merced / Colon-pathology / Aircraft)" — these specific datasets were not tested. The tested datasets are DTD, Fashion-MNIST, imagenet_val_top100, imagenet_val_top20. The claim wording refers to a specific named panel that doesn't match the experiment. This is a scope mismatch, and the plan's "cost-aware compression" provision only licenses testing 4-of-10 datasets, not substituting a different 4. The specific named datasets in the claim were never tested.

**Severity**: WARN

---

## Check F — Evaluation Type

**Finding**: real_gt for the substitute datasets (DTD, Fashion-MNIST, imagenet_val slices have real categorical labels). The evaluation is valid for the tested datasets; the scope issue is in Check E.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: warn**

Two WARNs: (1) The claimed specific panel (Birds/UC-Merced/Colon/Aircraft) was never tested; substitute datasets were used instead. (2) The claim scope includes a specific named dataset set that doesn't match the test set. However, the verdict (not-established: aligned strictly worse) is robust enough that it would likely hold even on the original panel. C4a is admitted to Stage 2 with WARN, but will be INTEGRITY_ONLY under the MAX_VERIFY_CLAIMS=1 cap.
