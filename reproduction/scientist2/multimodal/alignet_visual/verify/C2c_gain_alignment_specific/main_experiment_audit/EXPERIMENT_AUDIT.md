# Experiment Audit — C2c: Spearman Gain is Alignment-Specific

**Claim**: The multi-level Spearman gain from C2a is specifically attributable to human-similarity alignment (the aligned student strictly beats a matched non-human-aligned soft-label control at α = 0.05), not to generic KD label smoothing.

**Scope**: M3+M5 — `runs/M3_aligned_dinov2/eval_things_multilevel.json` vs `runs/M5_control_dinov2/eval_things_multilevel.json`

---

## Check A — Ground-Truth Provenance

**Finding**: PASS. Same as C2a — human pairwise similarity from THINGS training triplets; triplet accuracy from THINGS held-out. No model-derived GT. The specificity control (M5) uses unaligned SigLIP-So400m embeddings as the teacher signal, but the EVALUATION ground truth remains the THINGS human triplet labels.

**Severity**: PASS

---

## Check B — Score Normalization

**Finding**: PASS. Same Spearman computation as C2a/C2b. No normalization by model max/mean.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS. 
- M3 spearman_aggregate=0.5554 (verified in C2a)
- M5 spearman_aggregate=0.2736 (cited as 0.274 — matches `runs/M5_control_dinov2/eval_things_multilevel.json` on-disk)
- M4 (unaligned) spearman_aggregate=0.1891 (verified)
- Δ(M3−M4)=+0.366; Δ(M5−M4)=0.2736−0.1891=+0.085 (cited as +0.085 ✓)
- Human-specific component = M3−M5 = 0.5554−0.2736=+0.281 (cited as +0.281 ✓)
- 77% of total gain: 0.281/0.366=76.8% ≈ 77% ✓

All numbers verified on disk.

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. Same code as C2a/C2b for M3 and M5 evaluation. Both use `eval_student_similarity.py` run on different checkpoints. No dead code.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: PASS. The matched-cost control design is sound: M5 uses the exact same training schedule (lr=5e-5, batch=64, 1 epoch, 40k images, seed=42) but with an unaligned SigLIP-So400m teacher (no THINGS-fitted head). The claim's "matched non-human-aligned soft-label control" is precisely what M5 implements. No scope over-claim.

**Severity**: PASS

---

## Check F — Evaluation Type

**Finding**: real_gt — same as C2a.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: pass**

All checks pass. C2c has a clean matched-cost control design, all numbers verified on disk, and no methodology issues. The 77% aligned-specific component is a strong quantitative finding that rules out generic KD-as-label-smoothing.
