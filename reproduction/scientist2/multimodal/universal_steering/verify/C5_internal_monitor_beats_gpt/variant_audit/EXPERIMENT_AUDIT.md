# Variant Experiment Audit — C5 model-swap-deepseek-r1-llama8b

## Scope
Variant: `model-swap-deepseek-r1-llama8b`
Variant directory: `verify/C5_internal_monitor_beats_gpt/variants/model-swap-deepseek-r1-llama8b/`

## Audit checks

### 1. Single-change constraint (model path only)
- Changed: `MODEL_PATH` from `Llama-3.1-8B-Instruct` to `DeepSeek-R1-Distill-Llama-8B`
- Frozen: all hyperparameters (batch_size=8, max_length=512, rfm_iters=3, dtype=bfloat16, seed=42, ridge=1e-2)
- Verdict: PASS

### 2. Test split reuse (fair comparison)
- Variant reads `runs/C5_monitoring/halueval_test.jsonl` (400 rows) and `runs/C5_monitoring/toxicchat_test.jsonl` (118 rows) — same rows as main experiment
- GPT-4o scores reused from `runs/C5_baselines/halueval_gpt4o_scores.npy` (400) and `toxicchat_gpt4o_scores.npy` (118) — length-aligned with test splits (verified)
- Verdict: PASS — no data leakage, no row mismatch

### 3. Label alignment for GPT-4o AUROC
- `y_te_pm = np.where(labels_int[n_tr+n_va:] > 0, 1.0, -1.0)` extracts test labels from `all_rows = tr + va + te` — correct ordering
- `probe_auroc(np.array([1.0]), gpt_scores.reshape(-1,1), y_te_pm)` — correct AUROC formula matching main experiment
- Verdict: PASS

### 4. Block selection (no test peeking)
- Block selection = `argmax(val_auroc)` over validation set, not test set
- Test AUROC evaluated only after block is fixed
- Verdict: PASS

### 5. Success predicate
- `internal_beats_gpt = best_internal_test > gpt4o_test_auroc` per benchmark
- `variant_claim_supported = all(internal_beats_gpt for each benchmark)` — strict AND
- Matches main experiment's predicate
- Verdict: PASS

### 6. Ground truth
- HaluEval labels: from `data.jsonl` field `label` (0=no hallucination, 1=hallucination) — dataset-provided GT
- ToxicChat labels: from `data.jsonl` field `label` (0=benign, 1=toxic) — dataset-provided GT
- No fabricated labels
- Verdict: PASS

### 7. Scope claim alignment
- Variant tests: "do DeepSeek-R1-Distill-Llama-8B internal activations beat GPT-4o on both HaluEval and ToxicChat?"
- Matches the C5 claim scope (model-agnostic internal monitoring)
- Verdict: PASS

## Overall verdict
**PASS** — All methodology checks pass. Variant evaluation is clean.

## Result (for robustness judgment)
- HaluEval: best_internal=0.9851 vs GPT-4o=0.6854 — internal_beats_gpt=True
- ToxicChat: best_internal=0.8579 vs GPT-4o=0.8822 — internal_beats_gpt=False
- variant_claim_supported=False
