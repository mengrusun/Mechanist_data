# Experiment Audit Report — Claim C1

**Date**: 2026-07-22
**Auditor**: self-review (llm-chat MCP unavailable — empty config in .mcp.json; same degradation as experiment stage per CLAIMS_LEDGER.md Open Items)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C1 — Scale-Dependent Emergence: personal and attributed belief exhibit distinct emergence patterns across pythia-{410m, 1b, 2.8b}
**Linked milestones**: M1

## Overall Verdict: PASS

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
- GT is loaded from the dataset JSONL files (`belief_core/{reality,believe_truth,follow_belief}.jsonl`).
- Each example in the dataset carries a `gold` field (dataset-provided correct continuation) and a `distractor` field (foil), loaded by `belief_utils.py:load_task()` (lines 69-67 in belief_utils.py).
- No model-generated GT anywhere in the M1 pipeline.
- Evidence: `scripts/belief_utils.py:53-68` (BeliefExample dataclass: `gold: str`, `distractor: str` loaded from `d["gold"]`, `d["distractor"]`); `scripts/m1_behavioral_eval.py:26-57` (main loop calls `load_task`, evaluates with `evaluate_task_accuracy`).

### B. Score Normalization: PASS
- Accuracy is computed as binary fraction: `correct_bit = (lp_gold > lp_distractor)`, accumulated into `correct / total`.
- Wilson 95% CI computed from binomial statistics using the standard Wilson score interval formula (belief_utils.py `evaluate_task_accuracy`).
- No division by model's own max/mean output. PPL is absolute, not relative to anything model-specific.
- Evidence: `scripts/belief_utils.py:191-235` (evaluate_task_accuracy — binary correct counting + Wilson CI).

### C. Result File Existence: PASS
- M1 result files verified against EXPERIMENT_RESULTS.md reported numbers:
  - `refine-logs/artifacts/behavioral/pythia-410m/personal_belief.json` exists; acc=0.8516... ≈ 0.852 ✓
  - `refine-logs/artifacts/behavioral/above_chance_gate.json` exists ✓
- All 9 (model, task) result files confirmed present (verified via `ls` of artifact directory).
- Tracker rows for M1 all show status `done` with matching acc values.
- Evidence: EXPERIMENT_TRACKER.md M1 section; artifact files at `refine-logs/artifacts/behavioral/{model}/{task}.json`.

### D. Dead Code Detection: PASS
- `continuation_logprob_batch()` → called in `evaluate_task_accuracy()` (belief_utils.py:206).
- `evaluate_task_accuracy()` → called in `m1_behavioral_eval.py` main loop.
- `wilson_ci()` / Wilson CI computation → called inside `evaluate_task_accuracy()`.
- No metric-related functions appear defined but uncalled.
- Evidence: `scripts/belief_utils.py:191-235`; `scripts/m1_behavioral_eval.py`.

### E. Scope Assessment: PASS
- 9 (model, task) cells tested: 3 models × 3 tasks, all FULL datasets (n=227/681/681).
- Claim language: "distinct scale-dependent patterns", "dissociated at smallest scale" — precisely scoped to the 3 scales tested. No "comprehensive" / "extensive" language used.
- The 3×3 matrix is the entire scope — all cells reported including the below-chance cell (pythia-410m × attributed, 0.457). No selective reporting.
- Evidence: EXPERIMENT_RESULTS.md M1 section; full 3×3 table reproduced verbatim.

### F. Evaluation Type: real_gt
- Ground truth is dataset-provided gold/distractor continuations from belief_core JSONL.
- Comparison: log P(gold|prompt) > log P(distractor|prompt) — no model-generated reference.
- Classification: **real_gt** (binary forced-choice using dataset-annotated answer pairs).

## Action Items
None — no integrity concerns identified.
