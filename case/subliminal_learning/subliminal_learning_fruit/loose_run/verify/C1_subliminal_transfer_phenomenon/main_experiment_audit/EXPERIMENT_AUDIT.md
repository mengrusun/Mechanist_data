# Experiment Audit Report — Claim C1

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation per SKILL.md)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C1 — Subliminal banana preference transfers from a LoRA-anchored Qwen-Image teacher to a Qwen-Image student via a judge-filtered non-banana channel (mean P(banana) gap ≥ 5 pp over both controls, ≥ 6/8 seeds, banana_residue = 0)
**Linked milestones**: M-PREP, M0 (M0.1–M0.8)

## Overall Verdict: WARN

## Integrity Status: warn

The evaluation methodology is sound and all result files exist with matching numbers. The single WARN is for the pre-registered decision-rule deviation: `compute_verdict.py` code path maps `residue > 0` → `"inconclusive"`, but the on-disk `M0_verdict.json` field `phenomenon_status` was overridden to `"conditional"` post-hoc by the agent. The override is explicitly documented (`original_algorithmic_verdict: "inconclusive"`, `override_rationale` field present) and is scientifically defensible — but it represents a deviation from the plan's §M0 decision rule.

## Checks

### A. Ground Truth Provenance: PASS

The experiment measures whether a generative model produces banana images. There is no external dataset-provided GT for this; the judge (gpt-5.4) is the accepted measurement instrument. Key findings:

- Judge prompt (qwen_common.py:77-80): 10-way single-word MCQ over 10 fruit labels — not biased toward banana and not derived from model output.
- Judge calibrated on 20 hand-labeled images with `judge_recall = 1.000` (above the 0.9 gate). Sanity labels come from `data/judge_sanity/labels.jsonl` (10 real banana images + 10 base-model-generated non-banana images; not derived from the experiment models' own outputs at eval time).
- P(banana) computed as `count(judge labels == "banana") / N_prompts` — no circular GT.
- qwen_common.py header explicitly acknowledges: "All GT labels for banana classification come from the gpt-5.4 vision judge... The judge is the *accepted measurement instrument* per task.md, and is calibrated against a hand-labeled `judge_sanity` set."

Evidence: `src/qwen_common.py:3-8`, `src/eval_student.py:135-150`, `data/judge_sanity/labels.jsonl`.

### B. Score Normalization: PASS

P(banana) = `banana_n / num_prompts` (simple fraction, e.g., teacher_seed42: 119/160 = 0.74375). No normalization by model's own max. The gap metric is a raw difference: `mean_teacher − max(mean_ctrl_A, mean_ctrl_B)` — neither numerator nor denominator is derived from prediction statistics of the evaluated model.

Evidence: `src/compute_verdict.py:120-137`, `runs/M0_7_eval/teacher_seed42/p_banana.json:{"banana_n": 119, "num_prompts": 160, "p_banana": 0.74375}`.

### C. Result File Existence: WARN

**Files exist and numbers match:**
- `runs/M0_verdict.json` ✓ — `phenomenon_status: conditional`, `mean_gap: 0.6421875` (matches EXPERIMENT_RESULTS.md "64.2 pp")
- All 8 teacher seed p_banana files exist (`runs/M0_7_eval/teacher_seed{42..49}/p_banana.json`) ✓ with values matching EXPERIMENT_RESULTS.md table to 3 significant figures
- All 8 ctrl_b seed p_banana files exist ✓
- `runs/M0_7_eval/ctrl_a/p_banana.json` ✓ — `p_banana: 0.0375` (matches reported 0.037)
- `data/prompts/preference_160.jsonl`: 160 lines ✓ (matches "full 160 prompts")
- `data/channel_final_v2/teacher_channel.jsonl`: 154 lines ✓ (matches "N=154 pairs")
- PNG persistence: `runs/eval_gen/teacher/seed42/` contains 160 PNGs ✓
- Tracker rows M0.1–M0 all marked "done" ✓

**WARN flag — verdict override deviation:**
- `runs/M0_verdict.json` contains `"original_algorithmic_verdict": "inconclusive"` alongside `"phenomenon_status": "conditional"`. This documents that `compute_verdict.py` computed "inconclusive" (triggered by `residue=2 > 0` at line 149-151 of compute_verdict.py), and the agent overrode it to "conditional."
- The override is documented in `M0_verdict.json:override_rationale` and flagged in `CLAIMS_LEDGER.md` Open Items ("M0 STRICT-RULE OVERRIDE").
- **Is the override defensible?** Yes: (a) the multi-pass filter (`judge_filter.py`) ran 5 drop-passes and converged to 0 residue per pass; a fresh rescan at verdict time found 2 new images at different indices — consistent with ~1.3% judge stochasticity; (b) the 64.2 pp effect is 13× the 5 pp bar, making the residue issue a bookkeeping concern not a signal threat; (c) the `EXPERIMENT_RESULTS.md` §M0.4 documents the pattern in detail; (d) task.md Hard Constraints explicitly flag this for verify to audit.
- **Audit assessment**: the override understates the evidence (the unambiguous signal would support "established" if not for the residue bookkeeping; "conditional" is more conservative than "established"). The deviation is from the pre-registered rule, not from honesty about the numbers.

### D. Dead Code Detection: PASS

All metric functions that exist are called:
- `eval_student.py:main()` → called for each of 17 arms; output exists in `runs/M0_7_eval/*/p_banana.json` (17 directories confirmed)
- `compute_verdict.py:main()` → output exists at `runs/M0_verdict.json`
- `judge_filter.py:main()` → output exists at `data/channel_final_v2/{teacher,ctrl}_channel.jsonl`
- `qwen_cfg_wrapper.pipe_with_cfg()` → called from `eval_student.py:109-119` and `gen_channel.py`
- `assert_cfg_ok()` → called at `eval_student.py:34`

No metric functions defined but never called were detected.

### E. Scope Assessment: PASS

Actual scope:
- **Seeds**: 8 teacher-arm students + 8 Ctrl-B students (plan requires >7) ✓
- **Eval prompts**: 160 preference prompts (plan requires ≥160) ✓
- **Training data**: 154 equal-N-matched pairs (full clean channel) ✓
- **Ctrl arms**: Ctrl-A (base student) + Ctrl-B (base-teacher channel student) ✓
- **Judge calibration**: 20 hand-labeled images, judge_recall=1.000 ✓

Language in EXPERIMENT_RESULTS.md: "mean gap 64.2 pp on 8/8 seeds" — factual, not overclaiming. The word "conditional" accurately qualifies the residue caveat. No language like "comprehensive," "exhaustive," or "definitive" that would overstate the scope. CLAIMS_LEDGER.md explicitly lists caveats (single-seed v1 teacher retrain, residue override).

### F. Evaluation Type

**Classification: synthetic_proxy** (appropriate and acknowledged for a generative task)

The gpt-5.4 judge is the only feasible measurement instrument for the question "does this generated image depict a banana?" — no external dataset provides human labels for model-generated fruit images. The judge is calibrated against 20 hand-labeled images (judge_recall=1.000). The JUDGE_PROMPT (qwen_common.py:77-80) is a neutral 10-way MCQ with no banana bias. This proxy is explicitly acknowledged throughout the codebase.

## Action Items

1. **Verify residue-override defensibility in Stage 2** (required by task.md and this audit): re-score a sample of 30 random teacher-channel images with an alternative judge prompt template to independently assess whether the 2/154 residue is judge stochasticity or a filter recall gap. This is exactly the suggested method-swap variant for C1 if C1 is picked in Phase 3.

2. **No code changes required** for the evaluation methodology itself — the numbers are real, the scripts are called, and the judge is calibrated. The WARN is purely for the documented verdict override.
