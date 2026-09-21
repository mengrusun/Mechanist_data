# Experiment Audit — C2: Group Aggregation Dominates Single-Agent; No Single Aggregation Dominates

**Claim**: C2 — Given per-agent probe scores, at least one group-aggregation function's AUROC materially exceeds the best single-agent probe (predicate a: delta >= 0.05) AND no single aggregation dominates >= 9/17 domains (predicate b: diversity).

**Milestone scope**: M2 (M2.1 aggregation training, M2.2 diversity test, M2 verdict). M1 results are upstream inputs.

**Audit date**: 2026-07-14

---

## Check A — Ground Truth Provenance

**Finding**: Same ground truth as C1. `condition ∈ {collusive, honest}` → `gt_vote ∈ {1, 0}` is a design attribute. All aggregation scripts (`train_aggregation.py`) read labels from the M1 probe JSON (`probe["labels"]`), which in turn was read from the same `activations.pt` payload (populated by `extract_activations.py` from `scen.get("gt_vote", 0)`). No LLM-derived GT in the aggregation pipeline.

**Verdict**: PASS

---

## Check B — Score Normalization

**Finding**: Closed-form aggregations (mean-pool, max-pool, best-of-N) operate on raw logistic-regression decision-function outputs from M1 probe — no normalization by model max/mean. Learned aggregations (attention-pool, concat-mlp) apply `StandardScaler.fit(X_tr.reshape(-1, d_model))` on the train-split activations — standardization by train-split statistics, not model-output statistics. AUROC scores are computed by `sklearn.metrics.roc_auc_score` with no self-referential scaling.

**Verdict**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: `runs/M2/verdict.json` exists. Key numbers cited in EXPERIMENT_RESULTS.md:
- `best_group_auroc = 0.69` (best-of-N) — matches `verdict.json`.
- `best_single_agent_auroc = 0.665` (M1 scenario-level mean-pool) — matches `verdict.json` (`best_single_agent_auroc: 0.665`).
- `delta = 0.025` — matches `verdict.json` (`delta_group_minus_single: 0.025`).
- `diversity_top_count = 8` (attention-pool wins 8/16) — matches `verdict.json`.
- All five `agg_*.json` files exist at `runs/M2/`.
- `runs/M2/diversity.json` exists (referenced by `verdict_m2()`).

**Verdict**: PASS

---

## Check D — Dead Code

**Finding**: `verdicts.py` `verdict_m2()` is the aggregation function; it is called with `--milestone M2`. It reads each `agg_*.json` via `glob.glob(args.agg_glob)`, calls `roc_auc_score` through the per-aggregation results, and computes both predicates. `aggregation_diversity.py` computes per-domain argmax and writes `diversity.json`. `train_aggregation.py` trains and evaluates each aggregation variant and writes per-domain AUROC to the output JSON. All functions are called.

Note: `train_aggregation.py` computes `per_domain_auroc_heldout` (dev+test pooled) for the diversity test and `per_domain_auroc` (test-only) in the same run. Both are written to the output JSON; the diversity script reads `per_domain_auroc_heldout` per the comment in `aggregation_diversity.py`. This is correct design (pooling dev+test to get ~5/domain instead of ~3/domain for the diversity test).

**Verdict**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: C2's predicate is stated precisely: `best_group_AUROC - best_single_agent_AUROC >= 0.05` AND `no single aggregation dominant in >= 9/17 domains`. EXPERIMENT_RESULTS.md does not overclaim scope. Caveats:
1. **Diversity test on 16/17 domains**: `loan-approval` domain had only one class in dev+test and was excluded from per-domain AUROC. The predicate is stated as 9/17 but was effectively tested on 16/17. This is reported openly (`16 of 17 domains have enough held-out data`). The predicate-b pass (attention-pool 8/16 < 9) is correctly computed on 16 domains; even if the 17th domain were included, attention-pool would need both to be its win domain AND the 9th to flip the verdict — borderline situation flagged in the results.
2. Same under-power caveat as C1 (49-scenario test split, 49 heldout scenarios for diversity).

**Verdict**: WARN — diversity test effectively on 16/17 domains (1 domain excluded for single-class dev+test); under-power on AUROC gap measurement at N=49.

---

## Check F — Evaluation Type

**Finding**: All five aggregations evaluated on the same held-out test split as M1. Closed-form aggregations (mean-pool, max-pool, best-of-N) require no additional training data — they operate on the already-trained M1 probe's cached per-agent scores. Learned aggregations (attention-pool, concat-mlp) train small heads on the 200-scenario train split and evaluate on the 49-scenario test split. Design-attribute ground truth throughout. Evaluation type: supervised binary classification, `real_gt`.

**Verdict**: PASS

---

## Overall Verdict

| Check | Verdict | Notes |
|-------|---------|-------|
| A — GT provenance | PASS | Same design-attribute GT as M1 |
| B — Score normalization | PASS | Train-split standardization; no self-referential scaling |
| C — Result file existence | PASS | All agg_*.json, diversity.json, verdict.json confirmed |
| D — Dead code | PASS | All evaluation functions called |
| E — Scope | WARN | Diversity test on 16/17 domains; under-power on delta gap |
| F — Evaluation type | PASS | Supervised binary, real_gt |

**overall_verdict: warn**

Methodology is sound; WARN from diversity test with 16 (not 17) valid domains and small-N delta measurement. No fake results or broken evaluation.
