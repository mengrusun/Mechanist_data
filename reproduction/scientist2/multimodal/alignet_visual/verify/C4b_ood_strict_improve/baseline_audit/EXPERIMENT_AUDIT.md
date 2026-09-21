# Experiment Audit — C4b: Aligned Strictly Improves OOD

**Claim**: The aligned DINOv2 ViT-B strictly improves OOD robustness over the unaligned baseline on every held-out split: BREEDS-{entity13, living17, non-living26, entity30} and ImageNet-A.

**Scope**: M8 — `code/eval_downstream.py`, `runs/M8_ood/results.json`

---

## Check A — Ground-Truth Provenance

**Finding**: WARN. The claim specifies BREEDS-{entity13, living17, non-living26, entity30} and ImageNet-A as the OOD evaluation panel. The main experiment used substitutes: breeds_super13, breeds_super26 (true subpopulation-shift OOD constructed on the fly from the BREEDS-modified hierarchy over ImageNet-val 50k), plus imagenet_val_20_easy, imagenet_val_20_hard, fashion_mnist_ood.

The two BREEDS substitute splits are described as "constructed on the fly — prototypes from half the leaf classes per super-group, queries from the other half" — this is a researcher-defined subpopulation shift, not the standard BREEDS entity13/living17 benchmark splits. The GT labels for the splits are real (derived from ImageNet-val class labels), but the BREEDS split construction departs from the standard protocol. The other 3 splits (imagenet_val_20_easy/hard, fashion_mnist_ood) are in-distribution 1-shot slices, not genuine OOD — and they don't match the claim's specified OOD benchmarks.

**Severity**: WARN — substitute panel includes non-OOD splits alongside researcher-constructed OOD splits; the plan's named BREEDS splits and ImageNet-A were not tested.

---

## Check B — Score Normalization

**Finding**: PASS. Top-1 accuracy; not normalized by model max/mean.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS. `runs/M8_ood/results.json` exists on disk and all cited numbers match:
- breeds_super13: aligned=0.435, unaligned=0.234, Δ=+0.201 ✓
- breeds_super26: aligned=0.314, unaligned=0.199, Δ=+0.115 ✓
- imagenet_val_20_easy: Δ=−0.272 ✓
- imagenet_val_20_hard: Δ=−0.394 ✓
- fashion_mnist_ood: Δ=−0.202 ✓

All p_positive values match (1.00 for BREEDS splits, 0.00 for others).

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. Same `eval_downstream.py` code path as M7; all 5-split × 2-student cells exercised.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: WARN. The claim requires strict improvement on EVERY of the 5 named splits (BREEDS entity13/living17/non-living26/entity30 + ImageNet-A). None of these specific benchmarks were tested. The substitute includes 2 true OOD splits (BREEDS super13/26) and 3 non-OOD substitutes. The "strict improvement on every split" predicate as stated is unverifiable because the named splits don't exist in the experiment. The EXPERIMENT_RESULTS.md correctly downgrades to "conditional" for this reason.

**Severity**: WARN

---

## Check F — Evaluation Type

**Finding**: real_gt for the tested datasets (all use real class labels from ImageNet-val or Fashion-MNIST). The OOD-ness of each split is debatable (the 3 non-BREEDS splits are in-distribution), but GT labels are genuine.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: warn**

Two WARNs: (1) the specific named OOD panel (BREEDS entity13/living17/non-living26/entity30 + ImageNet-A) was never tested; (2) 3 of 5 substitute splits are not genuine OOD, making the "strictly improves OOD on every split" predicate unverifiable from the main experiment data. C4b is admitted with WARN; will be INTEGRITY_ONLY under the MAX_VERIFY_CLAIMS=1 cap.
