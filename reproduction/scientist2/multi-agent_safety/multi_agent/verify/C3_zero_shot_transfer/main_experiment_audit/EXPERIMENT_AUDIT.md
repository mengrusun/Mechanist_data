# Experiment Audit — C3: Zero-Shot Transfer to Structurally Different Multi-Agent Scenarios

**Claim**: C3 — Frozen M1 probe + M2 best aggregation, applied without any retraining to NARCBench-Transfer (6 families) and NARCBench-Stego (1 family), achieves AUROC >= 0.65 on at least 4 of 7 transfer targets.

**Milestone scope**: M3 (M3.1 scenario generation, M3.2 activation extraction, M3.3 zero-shot application, M3 verdict).

**Audit date**: 2026-07-14

---

## Check A — Ground Truth Provenance

**Finding**: Same design-attribute ground truth as C1/C2. Transfer scenarios have `condition ∈ {collusive, honest}` → `gt_vote ∈ {1, 0}` embedded at generation time in `gen_scenarios.py`. `apply_probe_zero_shot.py` reads `payload["labels"]` from the extracted `activations.pt`. No LLM-derived GT. Reviewer cross-check confirmed no CRITICAL label-source issues (per EXPERIMENT_RESULTS.md header).

**Verdict**: PASS

---

## Check B — Score Normalization

**Finding**: `apply_probe_zero_shot.py` applies the frozen M1 probe (`probe_bundle["scaler"]` + `probe_bundle["clf"]`) to transfer activations. The scaler was fitted on the M1 train split — not on the transfer data, and not on the model's own output distribution. Zero-shot means the scaler and classifier weights are frozen from M1; the transfer data is merely passed through. This is the correct protocol for a zero-shot transfer test. No self-referential normalization by the transfer model's statistics.

**Verdict**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: `runs/M3/transfer_results.json` exists. Key results cited in EXPERIMENT_RESULTS.md:
- `per_family_auroc`: research-framing=0.913, news-narratives=0.903, pipeline-sabotage=0.889, knowledge-poisoning=0.737, market-manipulation=0.654, code-review=0.647, stego-blackjack=0.525 — all confirmed in `verdict.json` (`per_family_auroc` field).
- `n_passing = 5` (>= 0.65 for all except code-review and stego-blackjack) — confirmed in `verdict.json`.
- `supported = true` — confirmed.
- Ablation results (`transfer_results_mean.json`, `transfer_results_attention.json`, `verdict_mean.json`, `verdict_attention.json`) exist and support claims about mean-pool (5/7) and attention-pool (4/7) ablations.
- `runs/M3/cost.json` exists: gpu_ids=[5,6], gpu_hours=2.6, confirming the run happened.

**Verdict**: PASS

---

## Check D — Dead Code

**Finding**: `apply_probe_zero_shot.py` is called with `--probe-pkl runs/M1/probe_best.pkl --agg-json runs/M2/agg_best-of-N.json`. It applies the probe and aggregation, computes per-family AUROC via `roc_auc_score`, and writes `transfer_results.json`. `verdicts.py` `verdict_m3()` reads `transfer_results.json` and evaluates the predicate (>= 4/7 families at >= 0.65). All functions are live and called.

**Verdict**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: C3's predicate is precisely stated: >= 4 of 7 transfer families with AUROC >= 0.65. The actual result is 5/7, and the failure cases (code-review at 0.647, stego-blackjack at 0.525) are reported honestly with explicit explanations: code-review is 0.003 below threshold (borderline), stego-blackjack requires covert bit-channel decoding, which is outside the probe's design scope. EXPERIMENT_RESULTS.md does not claim 7/7 or use "comprehensive transfer." The claim is stated as "at least 4 of 7" and the result is 5/7 — no overclaim.

Minor scope note: the transfer families are modest in size (25-36 scenarios per family), giving per-family AUROC estimates with SD ~0.05-0.10 at these counts. This is mentioned as a limitation (market-manipulation 0.654 is borderline) but not overclaimed.

**Verdict**: PASS

---

## Check F — Evaluation Type

**Finding**: Zero-shot transfer evaluation with design-attribute ground truth. The frozen probe was trained on NARCBench-Core only; the transfer sets are held-out at the family level (no overlap with training data). Per-family AUROC is computed by `roc_auc_score(d["y"], d["s"])` after grouping by family. Evaluation type: supervised binary, `real_gt`, genuine zero-shot (no retraining on transfer sets).

**Verdict**: PASS

---

## Overall Verdict

| Check | Verdict | Notes |
|-------|---------|-------|
| A — GT provenance | PASS | Design-attribute GT, no LLM-derived labels |
| B — Score normalization | PASS | Frozen M1 scaler; no transfer-data normalization |
| C — Result file existence | PASS | All transfer_results.json, verdict.json confirmed; numbers match |
| D — Dead code | PASS | All evaluation functions called |
| E — Scope | PASS | Predicate precisely stated; failures reported honestly |
| F — Evaluation type | PASS | Genuine zero-shot, real_gt, per-family AUROC |

**overall_verdict: pass**

C3's methodology is clean. The experiment is well-designed, results are internally consistent, and the claim is not overclaimed.
