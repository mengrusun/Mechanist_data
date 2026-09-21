# Experiment Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Cross-Modal Covert Transfer of Unsafe Behavior via a Text-Only Teacher-Generated Channel
**Claim**: C1 — In the fixed Qwen3.5-9B → Qwen3.5-9B multimodal transfer setup and the exact task.md recipe, text-only tuned-teacher-generated filtered data causes a ≥3pp drop in QA_I accuracy vs BOTH Ctrl-A and Ctrl-B, per seed across all 3 pre-registered seeds {42, 123, 2026}.
**Linked milestones**: M0 (M0.Setup, M0.S0.a, M0.S0.b, M0.S1, M0.S2, M0.S3, M0.S4, M0.S5, M0.S6, M0.S8)

## Overall Verdict: PASS

*C1's experimental process is methodologically sound across all six integrity checks.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS

Ground truth is loaded from `QA_I-00000-of-00001.parquet`, column `"Correct Answer"` (a pre-defined dataset label, not model-derived). Evidence:
- `scripts/eval_qa_i.py` line 60: `"gold": str(row["Correct Answer"]).strip().upper()` — directly from dataset row.
- The gpt-5.4 judge is used as a *classifier* to map free-form model text to {CORRECT, INCORRECT, OTHER} against this gold label — it does not generate ground truth.
- `scripts/eval_qa_i.py` lines 152-157: `JUDGE_PROMPT_TMPL.format(gold_letter=it["gold"], ...)` passes the dataset-derived gold into the classification prompt.
- The judge and the evaluated model are from different families (gpt-5.4 vs Qwen3.5-9B), eliminating self-referential GT.

No synthetic reference from model outputs. **PASS.**

### B. Score Normalization: PASS

Accuracy computed as `n_correct / max(1, tot)` where `tot = len(all_recs)` (dataset items), and `n_correct = sum(1 for r in all_recs if r.get("verdict") == "CORRECT")`.
- `scripts/eval_qa_i.py` lines 197-209: denominator is the number of dataset items (133), not model-specific max/mean.
- `scripts/aggregate_m0.py` line 50: `acc_from_per_item` = `CORRECT count / N`, with N from the dataset.
- Bootstrap CI (`scripts/bootstrap_ci.py`) uses item-level resampling within seeds; denominator is item count.
- No metrics normalized by model output statistics. No suspicious 0.99+ scores (max per-arm acc = 0.797). **PASS.**

### C. Result File Existence: PASS

All result files for C1's M0 milestones exist and contain the claimed numbers:
- `results/M0_VERDICT.json`: verdict=PASS, per_seed_table present, gaps match tracker (seed42: gap_B=+0.3008, seed123: +0.2932, seed2026: +0.1579, all ≥ 0.03). File exists non-empty.
- `results/eval/Ctrl-A.summary.json`: n=133, complete=true, acc=0.7820. Matches tracker R002.
- `results/eval/treated_seed42.summary.json`: n=133, complete=true, acc=0.4962. Matches R040.
- `results/eval/treated_seed123.summary.json`: n=133, complete=true, acc=0.4887. Matches R041.
- `results/eval/treated_seed2026.summary.json`: n=133, complete=true, acc=0.5940. Matches R042.
- `results/eval/Ctrl-B_seed42.summary.json`: n=133, complete=true, acc=0.7970. Matches R043.
- `results/eval/Ctrl-B_seed123.summary.json`: n=133, complete=true, acc=0.7820. Matches R044.
- `results/eval/Ctrl-B_seed2026.summary.json`: n=133, complete=true, acc=0.7519. Matches R045.
- `results/bootstrap_ci.json`: mean=0.2506, CI=[0.1955, 0.3033]. Matches R060.
- `logs/judge_calibration_seed{42,123,2026}.json`: measurement_valid=true for all 3 seeds. Matches R050-R052.
- `logs/filter_report_seed{42,123,2026}.json`: audit_status=audited_safe, actual_unsafe=0 all seeds. Matches R020-R022.
- All tracker rows R000-R080 (except optional R070) are status=done.

Numbers are internally consistent: per_seed_table in M0_VERDICT.json matches individual summary files within floating-point precision. **PASS.**

### D. Dead Code Detection: PASS

All metric functions in eval scripts are called:
- `scripts/eval_qa_i.py`: `load_qa_i_items()` called at line 122; `render_multimodal_prompt()` called at line 138; `parse_judge_verdict()` called at line 157; `main()` called at line 227 (`if __name__ == "__main__"`). `overall_acc` written to summary.json and consumed by `aggregate_m0.py`.
- `scripts/aggregate_m0.py`: `acc_from_per_item()` called at lines 87-88; `mean_std()` called at lines 184-187; all fields used in the `report` dict written to `M0_VERDICT.json`.
- `scripts/judge_calibration.py`: `stratified_sample()` called at line 116; `parse_verdict()` called inside the loop; `acc()` called at lines 178-184; all fields written to calibration JSON and consumed by `aggregate_m0.py`.
- `scripts/common.py`: `JudgeCache`, `call_judge`, `load_qa_i_items` all imported and used. **PASS.**

### E. Scope Assessment: PASS

C1's scope is exactly "3 pre-registered seeds × full QA_I (133 items) × 3 arms." The experiment matches this without overclaiming:
- 3 seeds: {42, 123, 2026} — pre-registered, all completed (6 eval runs + judge calibration × 3).
- Full QA_I: all eval summaries show n=133, expected_n=133, complete=true — no subsetting.
- 3 arms: Ctrl-A (base), Ctrl-B (per-seed base-teacher SFT), treated (per-seed tuned-teacher SFT).
- Claim wording: "per seed across all 3 pre-registered seeds" — no "comprehensive" or "extensive" overclaim. The binary gate (≥3pp per seed) is specific and unambiguous.
- Boundary: QA_I has 133 items, which is modest; the plan acknowledges this and does not claim statistical power beyond seed-level consistency. Bootstrap CI width (~10.8pp) is properly reported. **PASS.**

### F. Evaluation Type: real_gt

Ground truth derives from `QA_I-00000-of-00001.parquet`'s `"Correct Answer"` column, which is a dataset-provided correct-option label (A/B/C/D style multiple-choice). The gpt-5.4 judge converts free-form model text to a CORRECT/INCORRECT/OTHER classification against this pre-defined label. Classification is not model-generated; it is dataset-provided GT with an LLM classifier wrapper. **real_gt.**

## Action Items

None. All six checks pass. C1's experimental evaluation methodology is methodologically sound, with real dataset GT, proper denominator in accuracy computation, verified file existence matching tracker claims, no dead metric code, well-scoped evidence matching the claim, and dataset-provided ground truth.
