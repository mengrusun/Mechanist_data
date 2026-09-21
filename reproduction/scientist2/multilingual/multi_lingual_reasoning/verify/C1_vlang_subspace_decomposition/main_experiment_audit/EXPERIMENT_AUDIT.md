# Experiment Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C1 — Hidden representations of Qwen-3-4B-Thinking decompose into a language-specific subspace V_lang (identifiable from a small multilingual probe set via SVD/mean-difference) and an approximately orthogonal language-agnostic residual.
**Linked milestones**: M1

## Overall Verdict: PASS

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
Labels are dataset-partition-derived language identities (language directory / language code from FLORES-200 or MGSM). In `load_flores_probe(lang=...)`, sentences for language "en" are loaded from the English FLORES-200 file; the label index `li` is the sorted position in `MGSM_LANGS`. Zero model outputs are used as ground truth in M1. This is valid real_gt.
_Minor methodological note_: label could reflect dataset-specific domain artifacts alongside pure linguistic features — a scientific caveat, not a provenance fraud issue.

### B. Score Normalization: PASS
`heldout_lang_acc=0.968` is macro-averaged accuracy of sklearn `RidgeClassifier` on held-out MGSM activations using 80/20 stratified split; denominator is count of held-out test samples per class. `complement_acc=0.491` computed identically on the orthogonal-complement projection. No normalization by model's own prediction statistics anywhere in the M1 pipeline.

### C. Result File Existence: PASS
All claimed artifacts verified to exist with matching numbers:
- `results/m1/best_early.npz`: `heldout_lang_acc=0.9681818` (claimed 0.968 ✓), `complement_acc=0.4909` (claimed 0.49 ✓), `principal_angle_median_cos=0.1133` (claimed 0.11 ✓), `n_probe=250`, `rank_r=16`, `seed=43` (all match ✓)
- `results/m1/n250_r16_early_s43.npz`: same values (original file, not only the "best" copy)
- `results/m1/m1_summary.csv`: 273 rows (270 grid configs + 3 "best" duplicates)
- `runs/M1_seed42/m1_grid.log`: "90/90 configs done for seed=42, elapsed=525.0s"
- `runs/M1_seed43/m1_grid.log`: "90/90 configs done for seed=43, elapsed=527.0s"
- `runs/M1_seed44/m1_grid.log`: complete

### D. Dead Code Detection: PASS
`extract_answer()`, `grade()`, `numeric_equal()` from `mgsm_eval.py` are not called by `m1_locate.py` or `m1_locate_grid.py`. These are used by M2/M3 generation scripts, so they are not dead code project-wide. The M1 pipeline calls `_fit_v_lang`, `_train_linear_classifier`, `_eval_classifier` — all called and their outputs appear in the .npz files. No deceptive dead-code scoring substitution.

### E. Scope Assessment: WARN
- Grid 5×6×3×3 = 270 configs completed ✓; 3 seeds ✓
- The `layer_group` search uses one representative layer per group (midpoints: early=6, mid=18, all_non_upper=14), not a full per-layer sweep within each group. This limits the claim's generalizability across layers.
- Partial verdict ("identifiability leg supported; complete-decomposition leg not") is correctly acknowledged; scope is adequate for the partial result but insufficient for a full universal decomposition claim.
- Not fraudulent, but represents a meaningful scope limitation on the orthogonality claim.

### F. Evaluation Type: real_gt
Dataset-partition language identity labels from FLORES-200 / MGSM; no model-generated references.

## Action Items
- [WARN-E] Scope limitation on layer sweep is noted as a caveat; no fix required for the partial verdict as reported.
