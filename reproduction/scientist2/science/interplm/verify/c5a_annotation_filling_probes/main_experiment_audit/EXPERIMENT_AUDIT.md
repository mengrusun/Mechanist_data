# Experiment Audit Report — Claim C5a

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C5a — mean_PR-AUC(SAE linear probe) > mean_PR-AUC(neuron linear probe) with paired-Wilcoxon p < 0.05, on held-out Swiss-Prot annotations.
**Linked milestones**: M5

## Overall Verdict: WARN
*This is C5a's integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Ground truth is Swiss-Prot per-residue binary concept membership labels — external, independently curated annotations. Not derived from model outputs.
- Evidence: `scripts/m5_annotation_filling.py` — parses `uniprot_sprot.dat.gz` via `parse_swissprot_annotations()`, builds binary residue labels via `build_residue_labels()`.

### B. Score Normalization: PASS
- PR-AUC is computed from precision-recall curve on held-out test labels vs. probe predictions. No division by model's own maximum or any prediction statistic.

### C. Result File Existence: PASS
- `runs/m5/wilcoxon.json` exists: n_concepts=30, statistic=276.0, pvalue=0.1909, mean_sae=0.5907, mean_neuron=0.5906.
- `runs/m5/summary.md` matches: Mean PR-AUC (SAE)=0.5907, Mean PR-AUC (neurons)=0.5906, p=0.191.
- EXPERIMENT_RESULTS.md reports these same numbers.
- Tracker row m5: status=done, notes mean PR-AUC SAE=0.591, neurons=0.591; p=0.19.

### D. Dead Code Detection: WARN
- The function `per_concept_pr_auc()` in `scripts/m5_annotation_filling.py` is defined with `LogisticRegression` (lines 91-111) but the main execution path in `main()` uses `SGDClassifier(log_loss)` directly (lines 236-244).
- `per_concept_pr_auc()` is not called in `main()`. This is a dead code instance.
- However, the actual probe fitting did execute via SGDClassifier in the main path, and the PR-AUC results are real. The dead function is a code quality issue, not a fraudulent result.
- Evidence: `scripts/m5_annotation_filling.py:91-111` (per_concept_pr_auc with LogisticRegression, never called); `scripts/m5_annotation_filling.py:236-244` (SGDClassifier in main).

### E. Scope Assessment: WARN
- Planned: 50 top-K concepts × 3 seeds; test on full 387k residues.
- Actual: 30/50 concepts (20 excluded due to no test positives in 25k test subsample); 25k/387k residues for test (~6.5%); SGDClassifier max_iter=30 (low convergence). All transparently disclosed.
- The probe-capacity comparison (SAE vs neuron on same label set) is valid at the achieved scope. The null result is honestly reported.
- Assessment: WARN (acknowledged scope/method downgrade), not FAIL.

### F. Evaluation Type: real_gt
- Swiss-Prot per-residue annotations as held-out labels.

## Action Items
- Remove dead `per_concept_pr_auc()` function or call it in main.
- Re-run with full 387k test residues and larger test subsample to restore 50/50 concepts.
- Increase `max_iter` to ≥200 in SGDClassifier for better convergence.
- Consider `LogisticRegression(liblinear, max_iter=200)` for more reliable optimization.
