# Experiment Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + inline evidence review; llm-chat MCP call executed and findings incorporated below
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C1 — Per-emotion mean|Δaccuracy vs neutral| ≤ format-perturbation noise floor AND per-item sign-consistency ∈ [0.4, 0.6] on Qwen3-14B/GSM8K, across 12 emotion×intensity × 2 wording sources
**Linked milestones**: M2, M2b

## Overall Verdict: WARN

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Evidence: `scripts/run_prefix_eval.py:34-43` — `load_gsm8k()` reads the dataset from disk at `/data/zhenqian/data/gsm8k` (HuggingFace `load_from_disk`); gold is extracted from the dataset's `answer` field via `re.search(r"####\s*(-?[\d,]+)", r["answer"])` at line 42. No model output used as ground truth.
- Details: Gold answers are from the GSM8K dataset standard answer field. Evaluation type: `real_gt`. The regex correctly extracts the canonical `####` separator format used in GSM8K. No self-reference or synthetic proxy GT.

### B. Score Normalization: PASS
- Evidence: `scripts/run_prefix_eval.py:351-353` — `acc = n_correct / len(results)` where `n_correct = sum(r["correct"] for r in results)`. Denominator is number of items, not a model output statistic.
- Details: Raw accuracy is reported. No normalization by model maximum or model-specific baseline. Per-item delta is computed against the neutral condition's fixed accuracy (0.820), not against any dynamic model-derived max.

### C. Result File Existence: WARN
- Evidence:
  - 26 M2 result files confirmed present at `runs/M2/gsm8k_<condition_id>.json` for all 26 conditions (verified via `ls runs/M2/*.json`). Numbers match EXPERIMENT_RESULTS.md: neutral acc=0.820, happiness_1_human=0.708 (Δ=-0.112), etc.
  - 8 M2b perturbation files confirmed present at `runs/M2b/<perturbation_id>.json` (implicit from tracker rows showing all done).
  - C1's pass criterion also requires: per-item sign-consistency ∈ [0.4, 0.6] for EVERY prefix. **EXPERIMENT_RESULTS.md explicitly states**: "Per-item sign-consistency was not computed for all cells due to time" — this metric was never computed, yet C1's verdict requires it for ALL 24 emotional conditions.
  - The noise-floor threshold (p90=0.0524) is correctly derived from M2b data and reported. This half of C1 is properly evidenced.
- Details: WARN because C1's measurable predicate (IDEA_REPORT.md, EXPERIMENT_PLAN.md) includes per-item sign-consistency as a REQUIRED condition (not optional). The experiment_results logs only the macro-accuracy threshold part and explicitly admits the sign-consistency was not computed. This means C1's evaluation is incomplete — the evidence supports only one of two required predicates.

### D. Dead Code Detection: PASS
- Evidence: `scripts/run_prefix_eval.py` defines `parse_gsm8k`, `build_gsm8k_prompt`, `vllm_generate`, `hf_capture_activations`, and `main`. All are called in `main()`. The correctness computation at lines 328-334 calls `parse_gsm8k` and computes `correct`. No dead metric functions identified.
- Details: All defined evaluation functions are called in the evaluation pipeline. The `hf_capture_activations` function is called when `not args.skip_activations and args.activations_out` (line 356). For M2b runs, `--skip_activations` is passed per the tracker, so activations are appropriately not captured for noise-floor runs.

### E. Scope Assessment: WARN
- Evidence: 26 conditions × 500 items = 13,000 evaluations done. However:
  - C1's claim statement says "per-item sign-consistency … for every prefix" — not computed.
  - 500 items from 1319-item GSM8K test — subset noted in plan as permissible and sufficient for ±4.5pp CI width. This is acceptable.
  - EXPERIMENT_RESULTS.md notes "partial" verdict with the caveat that individual prefix cells can cross the threshold but per-emotion means don't — this is a meaningful finding but the claim as stated requires sign-consistency which is missing.
- Details: WARN because the scope is sufficient for the accuracy portion of C1 but the sign-consistency predicate was not evaluated, leaving C1's evidence incomplete. The paper narrative should not claim C1 as "supported" without the sign-consistency check.

### F. Evaluation Type: real_gt
- Classification: `real_gt` — ground truth loaded from HuggingFace GSM8K dataset's `answer` field using the standard `####` separator format.
- No synthetic proxy, no self-supervised proxy.

## Action Items
1. **Compute per-item sign-consistency** for all 24 emotional prefix conditions vs neutral on the existing M2 per_item results (data already on disk in `runs/M2/gsm8k_<condition_id>.json`). Load `per_item` arrays, compute fraction_positive = mean(delta > 0) per condition, check whether ∈ [0.4, 0.6].
2. **Update C1 verdict** based on the combined result: if sign-consistency is also satisfied, C1 is "supported"; if some conditions fall outside [0.4, 0.6], C1 is "not-supported" at the per-prefix level.
3. WARN does not block verify Stage 2 — C1 advances to Phase 3-10 with an integrity warning.
