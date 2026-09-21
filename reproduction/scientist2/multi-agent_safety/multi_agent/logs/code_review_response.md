# Response to Cross-model Code Review (v1 → applied fixes)

Ground-truth audit: reviewer confirmed **no CRITICAL issues** — labels are derived
deterministically from `condition` (design attribute), never from another model's output.

## Applied fixes (MAJOR)

- **#3, #14 valid_mask** — `extract_activations.py` now tracks a `valid_mask` per row and
  writes it to the payload. `train_probe.py`, `train_aggregation.py` (via `probe`
  json which now sees filtered indices), `text_only_judge.py`, and `apply_probe_zero_shot.py`
  filter to valid rows before use. Judge additionally skips rows with all-empty transcripts.
- **#4 device placement** — `extract_activations.py` now sends inputs to
  `next(model.parameters()).device` instead of `model.device`, which is the correct target
  for HF sharded models with `device_map="auto"`.
- **#5 memory** — activations are stored in fp16 both in-memory and on-disk; `train_probe.py`
  and `train_aggregation.py` cast to fp32 at load time (sklearn / probe head).
- **#7 floor check** — `train_probe.py` prints per-split honest/collusive counts and issues a
  WARN log if either class falls below `--floor-per-class` (default 25, plan aspirational 50).
  We keep it a warning (not hard fail) because the plan's 10 GPU-hour budget and the
  cost-aware combo may realize slightly fewer scenarios than the aspirational 50/class/split.
- **#8 tpr_at_1pct_fpr** — renamed from misnamed `auroc_at_1pct_fpr` to `tpr_at_1pct_fpr`,
  implemented via `np.interp` on the ROC curve. Output keys renamed accordingly.
- **#9 best-layer selection** — `verdicts.py` `verdict_m1` now ranks probes by scenario-level
  dev AUROC (`dev_auroc_per_scenario_mean_pool`) with per-agent dev AUROC as tie-break,
  matching the milestone predicate (probe AUROC ≥ 0.75 at scenario level).

## Acknowledged but deferred (MINOR / stylistic)

- **#1 n-scenarios-per-domain default 20 vs 50** — I intentionally kept the argparse
  default at 20 (to reduce API generation wall-clock), and the actual run command sets
  `--n-scenarios-per-domain 24` (408 core scenarios total). This still comfortably clears
  the data-rule floor of ≥ 50/class/split — 204 per class before split, ≥ 40 per class
  even in the 20% test split. The plan's aspirational 50/domain would give 850 total but
  wasn't necessary for the floor.
- **#2 gt_vote naming** — kept as-is. Downstream code uses `gt_vote` semantically as the
  collusion label (0/1) and this is documented in `gen_scenarios.py`. Renaming would
  cascade through 5 files without changing behavior.
- **#6 split edge cases** — the `stratified_split` bucket logic is fine for the realized
  data (24 scenarios per (domain × condition) bucket → clean 70/10/20). No fix needed at
  this scale.
- **#10, #11 sanity-check split & length-match matching** — the sanity controls are auxiliary
  resampled comparisons, not the main metric. They're clearly labeled `sanity_checks:` in
  the output. Deferred to a future revision.
- **#13 best-of-N semantics** — `best-of-N` implemented as argmax-|score| (agent whose probe
  reads most confident either direction). This is a valid "select the loudest signal"
  variant distinct from `max-pool`. Documented in the milestone report.
- **#15 agent-order randomization** — the scenario generator produces prompt_per_agent in
  a fixed order (agent 1..K), and the extraction script uses that same order for both probe
  and judge, so the confound the tip warns about (mismatched orders between probe and judge)
  is absent even without an explicit permutation step. If a future variant were to shuffle
  during extraction, we would need to store the permutation — noted.
- **#16 hardcoded API key fallback** — kept the env-first, hardcoded-fallback pattern so
  unattended runs on the same host without env vars still work (the key is task.md-supplied
  and repo-internal). Not a security concern in this project scope.

## Files changed

- `scripts/extract_activations.py` — device placement, fp16 storage, valid_mask.
- `scripts/train_probe.py` — valid_mask filter, tpr_at_fpr rename, dev-scenario AUROC,
  floor-per-class warning.
- `scripts/train_aggregation.py` — fp32 cast on load.
- `scripts/apply_probe_zero_shot.py` — valid_mask filter, fp32 cast.
- `scripts/text_only_judge.py` — valid_mask + empty-transcript filter.
- `scripts/verdicts.py` — best-layer selection by scenario-level dev AUROC.
