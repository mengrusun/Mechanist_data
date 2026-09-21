# Experiment Audit Report — Claim C4

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6)
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C4 — acc(π_θ) > acc(neutral) AND acc(π_θ) > acc(e*) on held-out GSM8K, both with lower 95% CI > 0 across ≥ 3 seeds
**Linked milestones**: M7

## Overall Verdict: FAIL

*C4's experimental process was not carried out — M7 was descoped entirely due to budget exhaustion. There are no result files, no scripts run, no evaluation performed. The experimental process is broken because it does not exist.*

## Integrity Status: fail

## Checks

### A. Ground Truth Provenance: FAIL
- Evidence: M7 (all of M7a/b/c/d) was descoped — no experiment was run. `EXPERIMENT_TRACKER.md` rows for M7: all show `descoped` status and `not run` result.
- Details: No GT was used because no experiment ran. This is a FAIL because there is nothing to audit — the experimental process was not carried out at all. Per the `/auto-verify` rubric (and the orchestrator's explicit watch-out instructions), this maps to FAIL. No evidence exists to check.

### B. Score Normalization: FAIL
- Same reason — no experiment ran, no scores computed.

### C. Result File Existence: FAIL
- Evidence:
  - `runs/M7a/reward_table.parquet` — NOT present (descoped, not run).
  - `runs/M7b/policy_sft_seed*.pt` — NOT present.
  - `runs/M7c/policy_rl_seed*.pt` — NOT present.
  - `runs/M7d/eval_seed*.json` — NOT present.
  - EXPERIMENT_TRACKER.md confirms: M7a/b/c/d all `descoped`, all `not run`.
- Details: Zero result files exist for C4's linked milestones. The claim cites M7 as "verified by" but M7 produced nothing. FAIL.

### D. Dead Code Detection: FAIL
- `scripts/build_reward_table.py`, `scripts/train_emotionrl.py`, `scripts/eval_emotionrl.py` exist but were NEVER called. All M7 scripts are dead code relative to actual experiment execution.
- This is not a "function defined but never called" within a script — it is an entire milestone's worth of scripts that were never invoked.

### E. Scope Assessment: FAIL
- Scope = 0 items, 0 seeds, 0 policy runs. Required: 2000 train items × 13 prefixes (M7a), 3 seeds (M7b/c), 500 held-out items × 3 seeds (M7d). Nothing was run.

### F. Evaluation Type: N/A (no evaluation was performed)

## Action Items
1. **C4 is marked INCONCLUSIVE** by Phase 2 — M7 was never run so no verification is possible. The causal chain from FAIL integrity → INCONCLUSIVE final state is correct.
2. To run C4 in a future iteration: need to download `Llama-3.2-1B` (or use `Llama-3.2-3B-Instruct` as fallback per EXPERIMENT_RESULTS.md note), run M7a reward table (≥2.17 GPU-h), M7b SFT (≥0.6 GPU-h), M7c RL (≥0.6 GPU-h), M7d eval (≥0.13 GPU-h) — total ≥3.5 GPU-h needed. Not feasible in current ~0.92 GPU-h remaining budget.
