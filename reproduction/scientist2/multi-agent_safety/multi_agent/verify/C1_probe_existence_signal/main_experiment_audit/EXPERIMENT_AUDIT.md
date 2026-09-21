# Experiment Audit — C1: Existence of Internal Collusion Signal

**Claim**: C1 — On Qwen3-32B-AWQ, a lightweight linear probe on per-agent residual-stream activations detects collusion at AUROC materially exceeding a strong text-only judge baseline. Predicate: probe_AUROC >= 0.75 AND probe_AUROC - judge_AUROC >= 0.05.

**Milestone scope**: M1 (M1.1 scenario generation, M1.2 activation extraction, M1.3 probe training, M1.4 judge baseline, M1 verdict).

**Audit date**: 2026-07-14

---

## Check A — Ground Truth Provenance

**Finding**: Ground truth is `condition ∈ {collusive, honest}` mapped deterministically to `gt_vote ∈ {1, 0}` — a design attribute of the scenario, never derived from another LLM's output. The scenario generation script (`gen_scenarios.py`) embeds the condition in the scenario template; `extract_activations.py` reads `scen.get("gt_vote", 0)`; `train_probe.py` treats it as the binary label throughout. `EXPERIMENT_RESULTS.md` confirms a reviewer cross-check found no CRITICAL label-source issues.

**Verdict**: PASS — ground truth is not LLM-derived.

---

## Check B — Score Normalization

**Finding**: Probe scores are raw logistic-regression decision-function outputs (not divided by the model's own max or mean). `train_probe.py`'s `train_and_eval_probe()` standardizes inputs using `StandardScaler.fit(X_tr)` (train-split mean and std, not the model's output distribution), then calls `clf.decision_function()`. AUROC is computed by `sklearn.metrics.roc_auc_score`. No self-referential normalization is present anywhere in the pipeline.

**Verdict**: PASS — no score normalization by model's own statistics.

---

## Check C — Result File Existence (claim-scoped)

**Finding**: `runs/M1/verdict.json` exists and is non-empty. Key metrics cited in `EXPERIMENT_RESULTS.md`:
- `probe_test_auroc_per_scenario_mean_pool = 0.665` — matches verdict.json field `probe_test_auroc_per_scenario_mean_pool: 0.665`.
- `judge_test_auroc = 0.60` — matches verdict.json field `judge_test_auroc: 0.6`.
- `predicate_b_probe_beats_judge_by_0.05: true` (0.665 - 0.60 = 0.065 >= 0.05) — confirmed in verdict.json.
- `predicate_a_probe_ge_0.75: false` (0.665 < 0.75) — confirmed in verdict.json.
- Sanity checks in `probe_layer48.json` match EXPERIMENT_RESULTS.md: label-permute=0.4115, length-match=0.6418, topic-swap=0.6514.
- Layer-27 probe: `test_auroc_per_scenario_mean_pool = 0.75` confirmed in `probe_layer27.json`.

**Verdict**: PASS — result files exist and numbers are internally consistent.

---

## Check D — Dead Code / Uncalled Evaluation Functions

**Finding**: `verdicts.py` `verdict_m1()` is called in `__main__` with `--milestone M1`. It explicitly reads `probe_best.json` (the dev-best probe), calls `roc_auc_score`, computes both predicates, and writes `verdict.json`. `train_probe.py` runs all three sanity checks (label-permute, length-match, topic-swap) when `--sanity-checks label-permute,length-match,topic-swap` is passed (confirmed in `run_probes.sh`). The `text_only_judge.py` script computes AUROC via `roc_auc_score` and writes `test_only_judge.json`. No evaluation functions found to be defined but never called.

**Verdict**: PASS — all evaluation code is live and called.

---

## Check E — Scope Claims (claim-scoped)

**Finding**: C1's predicate is stated precisely: `probe_AUROC >= 0.75 AND probe_AUROC - judge_AUROC >= 0.05` on a held-out test split. EXPERIMENT_RESULTS.md does not make "comprehensive" or "extensive" claims. Two genuine scope caveats:
1. **Realized sample size**: 282 of the aspirational 408 core scenarios were generated (74%). The 49-scenario test set gives ~24-25 per class, below the plan's ideal 50/class/split floor. EXPERIMENT_TRACKER.md notes this as a WARN (floor=25 per code, test at floor). `EXPERIMENT_RESULTS.md` tags this as `[suspected under-power: ...]` per the under-power tagging policy — reported openly rather than suppressed.
2. **Layer selection instability**: dev-best layer (48) has the worst test AUROC; test-best layer (27) exactly hits the 0.75 predicate. This is a caveat on the verdict robustness, not a scope overclaim.

**Verdict**: WARN — scope is not overclaimed but sample-size under-power is real (dev split below 25/class floor). Tagged appropriately in EXPERIMENT_RESULTS.md. Claim C1's verdict is "partial / not-supported" with explicit underpower annotation.

---

## Check F — Evaluation Type

**Finding**: Supervised binary classification with design-attribute ground truth. AUROC is the primary metric, computed on held-out test scenarios. This is `real_gt` (design attribute, not a synthetic proxy or model-generated label). Three sanity controls (label-permute, length-match, topic-swap) confirm the probe is not spuriously using confounds.

**Verdict**: PASS — evaluation type is appropriate.

---

## Overall Verdict

| Check | Verdict | Notes |
|-------|---------|-------|
| A — GT provenance | PASS | Design attribute → gt_vote, no LLM-derived GT |
| B — Score normalization | PASS | StandardScaler on train split; no self-referential normalization |
| C — Result file existence | PASS | verdict.json + probe JSON files confirmed, numbers match |
| D — Dead code | PASS | All evaluation functions called end-to-end |
| E — Scope | WARN | Under-power: 49 test scenarios, dev split at 25/class floor; layer-selection instability with small dev set. Reported openly. |
| F — Evaluation type | PASS | Supervised binary, real_gt, three sanity controls |

**overall_verdict: warn**

The experiment methodology is sound; the WARN reflects a real under-power caveat (small dev/test sets) and layer-selection instability. Neither issue constitutes fake results or broken evaluation — they are honest limitations that bound the claim's certainty. C1 is admitted to Stage 2 with a WARN annotation.
