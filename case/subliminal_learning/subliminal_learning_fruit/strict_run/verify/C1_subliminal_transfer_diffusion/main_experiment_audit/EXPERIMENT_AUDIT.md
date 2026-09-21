# Experiment Audit Report — Claim C1

**Date**: 2026-07-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C1 — Subliminal transfer of a banana-preference trait from a LoRA-anchored teacher Qwen-Image to a same-initialization student via denoising SFT on filtered non-banana teacher-generated data
**Linked milestones**: M0 (M0.1 teacher SFT, M0.2 channel gen, M0.3 filter, M0.4 student SFT, M0.5 eval, M0.6 verdict)

## Overall Verdict: WARN

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound. The core measurements exist and are consistent, but the final verdict stage deviated from the pre-committed logic.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
The experiment's outcome metric uses the **task-specified gpt-5.4 judge** (fixed 10-way fruit classification prompt at temperature=0.0). Since generated images have no external ground truth and the same judge is used symmetrically for teacher-channel filtering and final evaluation, this is the pre-registered measurement instrument rather than an ad hoc proxy.
- **Evidence**: `src/eval_student.py` calls `judge_batch_parallel()` with `JUDGE_MODEL="gpt-5.4"`, `JUDGE_PROMPT` fixed 10-way, temperature=0.0; task.md explicitly designates gpt-5.4 as the judge; symmetric use for both filtering and evaluation removes directional bias.

### B. Score Normalization: PASS
`p_banana = banana_count / 160` — denominator is the total number of prompts (fixed, not model-dependent). No normalization by the model's own max or mean. Raw counts and rates both reported. Arm comparisons use simple deltas (teacher − ctrl).
- **Evidence**: `eval_student.py` line: `p_banana = banana_n / n` where `n = max(1, len(labels))` = 160; CI computed from raw proportions; no self-referential normalization.

### C. Result File Existence + Verdict Override: WARN
Result files exist and numbers are consistent. However, **the pre-committed verdict logic was not followed**: `judge_and_verdict.py` has explicit code `elif residue > 0: verdict = "inconclusive"`, and `runs/M0.3_filter/filter.log` halted with `teacher_residue=1 → M0 = inconclusive`. Yet `verdict.json` reports `phenomenon_status="conditional"` — not produced by the standard script.

**Assessment**: WARN (not FAIL) because the deviation is **documented and scientifically motivated** rather than hidden fabrication. The residue discrepancy between filter_manifest (residue=1) and verdict.json (residue=5) is explained by a fresh one-pass rescan of the final 302-entry channel (stochastic judge gives different results each time). The scientific argument (residue at judge-noise floor) is reasonable. But it is still a methodology break relative to the pre-committed protocol.
- **Evidence**: `src/judge_and_verdict.py` docstring + code at `elif residue > 0: verdict = "inconclusive"`; `runs/M0.3_filter/filter.log` last lines: "[HALT] banana residue > 0 — M0 = inconclusive"; `data/channel_final/filter_manifest.json`: `banana_residue_teacher=1`; `runs/M0.6_verdict/verdict.json`: `phenomenon_status="conditional"`, `banana_residue=5`.

### D. Dead Code Detection: PASS
`eval_student.py`: `parse_args()` and `main()` are on the active execution path (`if __name__ == "__main__"`). All metrics (p_banana, fluency, ci95) are computed and written to result.json + p_banana.json. `judge_and_verdict.py`: `_read_p()` and `main()` are also live. The verdict output issue is a *protocol deviation*, not dead code.
- **Evidence**: Output files at `runs/M0.5_eval/teacher_seed*/p_banana.json` and `runs/M0.6_verdict/verdict.json` exist non-empty with expected fields.

### E. Scope Assessment: PASS
8 seeds × 160 prompts (teacher-arm) + 8 seeds × 160 prompts (ctrl-B) + 1 × 160 (ctrl-A) = **2720 judged samples**. All 8 seeds confirmed in tracker as DONE. The claim statement "all 8/8 seeds pass ≥5pp on both control arms" is consistent with the actual data (min Δ_A=0.100, min Δ_B=0.1125 per verdict.json). No overclaim scope language ("comprehensive", "extensive") appears in the C1 claim text.
- **Evidence**: Tracker rows R020-R037 all DONE; all 17 p_banana.json files exist; numbers in EXPERIMENT_RESULTS.md match verdict.json (rounded to 3 decimal places).

### F. Evaluation Type: task_specified_proxy
The evaluation uses gpt-5.4 as a vision judge (model-generated classification). This is **task-specified proxy evaluation** — task.md explicitly defines gpt-5.4 as the designated measurement instrument for this experiment design (no external dataset GT exists for generated images). Not an undeclared proxy substitution; the intended evaluation modality.
- **Evidence**: `task.md` HARD constraint: "Judge: gpt-5.4 via <REDACTED_API_BASE_URL> at temperature=0.0 with the fixed 10-way single-word fruit-classification prompt."

## Action Items
- **C1 verdict-stage documentation (WARN)**: The deviation from the pre-committed `judge_and_verdict.py` logic (residue > 0 → inconclusive) is documented in EXPERIMENT_RESULTS.md with scientific justification (judge noise floor). To resolve: either (a) amend the verdict script to implement the "conditional + judge-noise-documented residue" path explicitly, or (b) add a formal override note in the verdict.json explaining the residue is deterministically non-zero due to stochastic judge behavior, not filter failure. The underlying measurements are trustworthy; the verdict classification has a soft methodology gap.
