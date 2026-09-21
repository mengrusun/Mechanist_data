# Experiment Audit Report — Claim C2

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via API) + executor verification
**Project**: Cross-Modal Covert Transfer of Unsafe Behavior via a Text-Only Teacher-Generated Channel
**Claim**: C2 — Some low-rank residual-stream direction in the student's language tower (or a small set of concentrated LoRA A rows in the fallback branch) causally mediates the covert-channel safety drop, with sign + monotone dose-response + specificity all confirmed; OR the mechanism arc reports a BOUNDED NULL under this ontology.
**Linked milestones**: M1 (M1.L0, M1.L-Core), M2 (M2.2a, M2.2b, M2.2b.aggregate, M2.2c, M2.2d)

## Overall Verdict: WARN

*External LLM reviewer returned WARN (max severity WARN across checks A–F). Executor post-hoc verification of result-file existence and call-chain resolves the ambiguity in Checks C and D to PASS, but the external reviewer's WARN is preserved as the official audit verdict per the reviewer-independence protocol.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS

C2's evaluation reuses the same QA_I infrastructure as C1 (M0). Ground truth is `QA_I-00000-of-00001.parquet` column `"Correct Answer"` (dataset-provided multiple-choice labels). The gpt-5.4 judge classifies model text against this pre-defined label — it does not generate ground truth. Judge and evaluated model are different families. The mechanism tests (ablation, steering, specificity) measure accuracy changes against the same GT. **PASS.**

### B. Score Normalization: PASS

Accuracy is n_correct / n_items (133). The recovery fraction is `(Acc_ablated - Acc_treated) / (Acc_Ctrl-A - Acc_treated)` — a signed fraction of the gap, not a self-normalization. Spearman ρ is computed over the 7-point α grid, not normalized by model statistics. σ_proj scaling is a perturbation convention for α calibration, not a metric normalization. **PASS.**

### C. Result File Existence (C2-scoped): PASS

*External reviewer flagged WARN due to incomplete artifact traceability in the prompt. Executor verification confirms all files exist with cited keys:*

- `results/mech/M1_l_core.json`: top_k_layers=[31,30,29], stability_gate_result=PASS, norm_v seed42 top3=[172.9, 165.2, 155.3] — matches EXPERIMENT_RESULTS.md.
- `results/mech/M2_2a_seed{42,123,2026}.json`: recovery_fraction key present; values +0.079, +0.256, -0.120 — match tracker R200-R202 and EXPERIMENT_RESULTS.md Table.
- `results/mech/M2_2b_alpha{-2,-1,-0.5,0,0.5,1,2}_seed{42,123,2026}.json`: all 21 files exist; acc_steered and alpha_actual keys confirmed. Values match EXPERIMENT_RESULTS.md M2.2b table.
- `results/mech/M2_2c_randdir_alpha*_seed*.json`: all 21 files exist; random_direction=True confirmed.
- `results/MECHANISM_VERDICT.json`: verdict=BOUNDED_NULL, recovery_pass=false, median_rho=+0.371, spearman_rho_per_seed and matched_random_mean_abs_delta_per_seed keys all present — match EXPERIMENT_RESULTS.md.
- `launch_m1_m2.sh`: explicitly calls `aggregate_mechanism.py` with correct glob arguments → `--out $ROOT/results/MECHANISM_VERDICT.json`.

All cited numbers verified to exist at claimed paths with correct keys. **PASS** (executor override of reviewer WARN).

### D. Dead Code Detection: PASS

*External reviewer flagged WARN (no call graph provided in prompt). Executor verification confirms:*

`launch_m1_m2.sh` runs all stages sequentially: `stage_l0()` → `stage_lcore()` → `stage_m2_2a()` → `stage_m2_2b()` → `stage_m2_2c()` → `stage_m2_2d()`. Each stage explicitly calls the relevant script:
- `l_core.py` called in `stage_lcore()` → `results/mech/M1_l_core.json`
- `ablate_and_eval.py` called in `stage_m2_2a()` × 3 seeds → `M2_2a_seed{s}.json`
- `steer_and_eval.py` called in `stage_m2_2b()` × 21 runs → `M2_2b_alpha{a}_seed{s}.json`
- `steer_and_eval.py --random_direction` called in `stage_m2_2c()` × 21 runs → `M2_2c_randdir_alpha{a}_seed{s}.json`
- `aggregate_mechanism.py` called in `stage_m2_2d()` → `results/MECHANISM_VERDICT.json`

All helper functions (`compute_sigma_proj`, `make_add_hook`, `make_projout_hook`, `install_hooks`, `eval_with_hooks`) are imported and called within the invoked scripts. **PASS** (executor override of reviewer WARN).

### E. Scope Assessment: PASS

C2 is explicitly disjunctive ("causally mediates ... OR the mechanism arc reports a BOUNDED NULL"). The actual verdict is BOUNDED_NULL — within the scope of the claim. Seeds: 3 pre-registered {42, 123, 2026}. Items: full 133 QA_I. Off-target (R420-R422) skipped — plan explicitly states "DELETE this milestone if unusable; do NOT invent a benchmark" (M2.2c `--off-target-abort-if-unusable true`). This is not a scope failure. Claim wording does not use "comprehensive" or "extensive" overclaiming language. **PASS.**

### F. Evaluation Type: real_gt

Ground truth from `QA_I-00000-of-00001.parquet` `"Correct Answer"` column (dataset-provided labels). The mechanism tests measure accuracy changes against the same GT. **real_gt.**

## Action Items

None blocking. Checks C and D were flagged WARN by the external reviewer due to incomplete context in the audit prompt, but executor verification confirms full artifact traceability and no dead code. The overall verdict is WARN due to the external reviewer's response, honoring reviewer-independence protocol.
