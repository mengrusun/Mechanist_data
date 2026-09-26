# Experiment Audit — Variant: model-swap-judge-gpt4o (C1)

**Audit date**: 2026-07-10
**Variant**: `C1/model-swap-judge-gpt4o`
**Dimension**: model (eval judge swap: gpt-5.4 → gpt-4o)
**Auditor**: auto-verify Phase 9 (variant integrity gate)

## Summary

**Overall verdict**: PASS

The variant swaps only the judge model while keeping every other experimental element identical (same student checkpoints, same eval dataset, same verdict aggregation logic, same gold-truth source). All six methodology checks pass — the judge-swap does not introduce GT fabrication, score normalization artifacts, phantom results, dead code, scope violations, or evaluation type changes.

---

## Checks

### A. Ground-truth provenance

**Verdict**: PASS

Gold truth still sourced exclusively from QA_I dataset's own `Correct Answer` column. The judge (now gpt-4o) is used as a semantic string-matcher — adjudicating whether the model's generated answer matches the gold letter/text — not as a GT generator. Unchanged from main experiment.

### B. Score normalization

**Verdict**: PASS

Accuracy = n_correct / n_total computed by `scripts/m0_verdict.py` (unchanged). Drop = Acc(Ctrl) − Acc(treated). No normalization by model-specific max or mean. gpt-4o judge output format matches the expected {CORRECT, INCORRECT, OTHER} labels (confirmed by 133-row eval_ctrl.jsonl).

### C. Result file existence

**Verdict**: PASS

All four arm output files exist and have 133 rows each:
- `verify/C1_cross_modal_subliminal_transfer/variants/model-swap-judge-gpt4o/eval_ctrl.jsonl` — 133 rows
- `eval_seed100.jsonl` — 133 rows
- `eval_seed200.jsonl` — 133 rows
- `eval_seed300.jsonl` — 133 rows
- `result.json` — acc_ctrl=0.7895, per-seed drops: {100: +24.06 pp, 200: +15.04 pp, 300: -3.01 pp}, verdict=conditional

Numbers internally consistent: acc_ctrl_n_correct=105/133=0.7895; seed100: 73/133=0.5489; seed200: 85/133=0.6391; seed300: 109/133=0.8195.

### D. Dead code / dead metric

**Verdict**: PASS

`scripts/m0_verdict.py` is the same verdict aggregation script as the main experiment. All invoked code paths (acc_of, paired_bootstrap_ci, load_jsonl) are confirmed live. The `--skip_aux` flag was passed intentionally (auxiliary robustness checks paraphrase/decoding are main-experiment-only; omitting them from the variant eval is correct because the robustness probe here is judge calibration, not generation sensitivity).

### E. Scope

**Verdict**: PASS (no new scope tension introduced)

The variant does not change the claim being tested (C1 requires per-seed unanimity ≥3 pp). The scope tension already flagged in the baseline audit (Check E: WARN — claim predicate requires unanimity, main experiment found conditional) is inherited unchanged. No new overclaim.

### F. Evaluation type

**Verdict**: PASS

Evaluation type remains `real_gt` (benchmark-sourced gold answers). The gpt-4o judge is used as a semantic comparator against real gold, not as a label provider. The variant cache `caches/eval_cache_gpt4o.jsonl` is separate from the main experiment cache, preventing contamination. Judge API calls routed through the same endpoint (https://www.dmxapi.cn/v1).

---

## Phase 9 integrity verdict for this variant

`combined = max_severity(experiment=PASS, mechanism=n/a) = PASS`

This variant is **integrity-eligible** — it enters the robustness numerator/denominator.
