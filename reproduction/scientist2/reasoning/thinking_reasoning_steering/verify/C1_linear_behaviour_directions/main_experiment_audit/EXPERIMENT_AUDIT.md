# Experiment Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C1 — Each reasoning behaviour maps onto an approximately linear direction in the residual stream: held-out linear-probe ROC-AUC ≥ 0.75 AND first-PC alignment |cos(v_b(L*), first_PC)| ≥ 0.7.
**Linked milestones**: M1

## Overall Verdict: WARN
*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
The probe labels come from a labelled contrastive corpus, but the behaviour annotations are LLM-judge derived (GPT-5.4, T=0) rather than human/dataset ground truth. This is an explicit proxy label pipeline, not model-self-derived labels, so it does not constitute a hard FAIL under the checklist. However, it is not true external GT either — the same model (GPT-5.4) that generates the benchmark tasks is also the annotator. The kappa scores (κ = 0.77–1.00) are reported for judge stability (inter-run, not inter-annotator) and are within acceptable range. WARN for absence of human validation.

### B. Score Normalization: PASS
ROC-AUC is computed against 0/1 probe labels. First-PC alignment is a cosine similarity between mean-diff direction and first right-singular vector of paired activation diffs. No metric is normalized by the model's own prediction statistics. All values are raw — no forbidden denominator.

### C. Result File Existence: PASS
The reported AUC values in EXPERIMENT_RESULTS.md (0.977, 0.840, 1.000, 0.892) are consistent with the M1 result structure (runs/M1_locate/results.json). The first-PC alignment values (0.263, 0.364, 0.340, 0.447) are present and stored in per_behaviour entries. The sigma_proj values are stored per behaviour. Direction files v_{behaviour}_L{L*}.pt exist per the M1 output. Numbers cited are plausible and consistent.

### D. Dead Code: PASS
`first_pc_alignment()` is called in run_M1_locate.py and its result stored in `results.json` under `first_pc_align`. `compute_sigma_proj()` is called and `sigma_proj_at_L_star` stored. `train_probe_per_layer()` is called and ROC-AUC stored per layer. All metric functions defined in M1 are actively called and their results appear in the output file.

### E. Scope Assessment: PASS
M1 tests 200 chains across 4 behaviours with all 32 layers swept. The report is honest that the ROC-AUC criterion (≥ 0.75) is met for all four behaviours while the first-PC alignment criterion (≥ 0.70) fails for all four. No overclaiming language ("comprehensive", "extensive") is used inappropriately — the claim's partial/fail verdict is explicitly reported. Backtracking L*=1 pathology is noted as a caveat.

### F. Evaluation Type: synthetic_proxy
Behaviour labels are proxy annotations from an external LLM judge (GPT-5.4, T=0) rather than dataset-native or human GT. The probe's training labels are LLM-annotated. However, the ROC-AUC computation itself is clean (binary 0/1 labels vs. probe probabilities). The taxonomy is frozen (version=1) and reused across all milestones.

## Action Items
- Consider inter-annotator agreement study with human annotators on a small subset (e.g., 20 chains) to validate LLM-judge taxonomy alignment.
- The kappa statistic reported (κ=0.77–1.00) measures inter-run stability of the LLM judge, not true inter-annotator agreement — this distinction should be explicit in any paper write-up.
