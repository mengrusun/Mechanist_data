# Experiment Audit Report — Claim CM

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + evidence review
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: CM — Some low-rank residual-stream direction on Qwen3-14B carries the emotion identity (Location) and causally modulates per-item GSM8K Δaccuracy (Intervention), with matched-length filler controls null
**Linked milestones**: M5, M6

## Overall Verdict: WARN

*CM's experimental process is partially sound. M5 (Location) is clean and trustworthy. M6 (Causal) has significant scope gaps (4/13 planned runs descoped; only 1 emotion tested; specificity controls not run) but the actual runs were executed with real GT and no methodological fraud. The main integrity concern is scope/completeness, not GT provenance or score normalization.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Evidence:
  - M5 (probe): `probe_location.py` uses cached activations from M2 runs; emotion identity labels come from `prefixes.json` (human-defined emotion metadata), not from model outputs. The label source is `EMOTIONS.index(meta["emotion"])` at `probe_location.py:119` where `meta = cond_meta.get(cid)` from the `prefixes.json` file — a human-authored metadata file.
  - M6 (causal): `mechanism_causal.py` uses GSM8K gold from `load_gsm8k()` (dataset-provided, same as M2). Gold at line 302: `gold = it["gold"]` from the loaded item dict.
  - No model output used as GT for either M5 or M6.
- Details: real_gt for both sub-milestones.

### B. Score Normalization: PASS
- M5: probe accuracy = `clf.score(Xte, yte)` at `probe_location.py:52` — sklearn's proportion-correct, denominator is n_test_items.
- M6: accuracy = `n_correct / n_valid` at `mechanism_causal.py:336`. No model-derived denominator.
- Details: Neither M5 nor M6 uses model-output-derived normalization.

### C. Result File Existence: WARN
- M5: `reports/M5_location.json` and `reports/M5_directions.pt` exist per tracker (status=done). Probe acc=1.00 at layers 4-36, null~0.28. Consistent with EXPERIMENT_RESULTS.md.
- M6: 9 of 13 planned runs exist in `runs/M6/`. Missing 4 runs: `M6_patch_filler_L4`, `M6_patch_filler_L8`, `M6_steer_medqa_L8_ap1`, `M6_steer_L20_null_ap1` — all descoped per tracker.
- CM's claim requires (i) steering dose-response monotone on ≥3/4 sites, (ii) filler-control Δ ≈ 0, (iii) off-target near zero, (iv) patching sign matches C1. Only the dose-response runs are complete; specificity controls (ii, iii) are MISSING. The off-layer null check (iv companion) is also missing.
- Also: only `happiness_2_human` was tested as the emotion_pair for all M6 runs — 1 of 12 planned emotional prefixes. The patching sufficiency test (12 emotional prefixes × 2 sites) was reduced to 1 prefix × 2 sites.
- WARN: significant scope gaps, not phantom results. The runs that were performed are real.

### D. Dead Code Detection: PASS
- `mechanism_causal.py` does NOT have dead functions. `mechanism_causal_batch.py` (referenced in tracker) is an alternate batch-mode wrapper; the single-item `mechanism_causal.py` was used for actual runs. Both scripts define `main()` which is called. No metrics defined but never called.
- `probe_location.py`: `probe_layer`, `probe_length_shuffled`, `load_activations` all called in `main()`.

### E. Scope Assessment: WARN
- M5: 200 items × 24 conditions × 10 layers = probe dataset adequate. 3 seeds. Result is robust (1.00 probe vs 0.28 null — large gap, reproducible). Scope adequate for Location sub-claim.
- M6: 50 items (vs 200 planned), 1 emotion (vs 12 planned for sufficiency), 0/3 specificity controls run, 1 off-layer check not run. CM's causal sub-claim is significantly under-powered.
- The EXPERIMENT_PLAN.md explicitly tags M6 with `method_sensitive: [n_pairs, sites, metric, gpu_hours]` — all four flagged fields were re-bound downward due to budget.
- WARN: Location sub-claim is adequately scoped. Causal sub-claim is inadequately scoped, but the runs that exist are honest. Overall verdict WARN (not FAIL) because no fraud occurred — the scope was documented and the descoped controls are listed explicitly in the tracker as "not run."

### F. Evaluation Type
- M5: probe training/evaluation on cached activations — intrinsic evaluation (no external GT needed for probing; the "GT" is the emotion label from the prompt corpus, which is human-authored).
- M6: `real_gt` for GSM8K accuracy measurement.

## Action Items
1. Run the 4 descoped M6 controls (filler patch at L4/L8, off-target MedQA steering, off-layer L20 null) to complete CM's causal sub-claim. Estimated cost: ~0.6 GPU-h (below the ~0.92 GPU-h remaining budget).
2. Run ≥2 additional emotional prefixes through the patching sufficiency test (cross-emotion patch, not just happiness→happiness identity patch).
3. WARN does not block Stage 2 — CM advances to Phase 3-10 with an integrity warning.
