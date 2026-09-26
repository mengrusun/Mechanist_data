# Variant Diff — model-swap-judge-gpt4o

**Claim**: C1 — Cross-Modal Subliminal Transfer
**Dimension**: model (eval judge model swap)
**Swap**: gpt-5.4 → gpt-4o (via same dmxapi.cn API endpoint)

## What changed vs. the main experiment

1. **Judge model**: `gpt-4o` instead of `gpt-5.4`. All judge prompts are identical (same ANSWER_JUDGE_TEMPLATE from scripts/qa_i_eval.py). Same API endpoint (https://www.dmxapi.cn/v1).
2. **Judge cache file**: `caches/eval_cache_gpt4o.jsonl` (separate from main-experiment `caches/eval_cache.jsonl`) to avoid contamination.
3. **Output files**: `verify/C1_cross_modal_subliminal_transfer/variants/model-swap-judge-gpt4o/eval_{ctrl,seed100,seed200,seed300}.jsonl`

## What is NOT changed

- Student model checkpoints: `ckpts/student_seed{100,200,300}/` (same as main experiment — no retraining)
- Base model: `/mnt/quarkfs/share_model/Qwen3.5-9B`
- Eval dataset: `data/QA_I-00000-of-00001.parquet` (133 items)
- Frozen 80/20 split: `results/qa_i_split.json`
- Safety-relevance labels: `results/safety_relevance_labels.json`
- Generation: greedy decoding, bs=32, enable_thinking=False, max_new_tokens=256
- Judge prompt template: identical ANSWER_JUDGE_TEMPLATE
- Verdict aggregation: same m0_verdict.py logic (count CORRECT per judge output)

## Hyperparameter adjustments

None required — the judge-model swap does not require any hyperparameter changes. The student generation (model forward pass) is identical. Only the judge API call changes.

## Why this tests the claim

C1's primary metric (Acc(QA_I)) depends on whether gpt-5.4 labels model outputs as CORRECT/INCORRECT/OTHER. If the ~23 pp / ~15 pp drops on seeds 100/200 are artifacts of gpt-5.4's specific calibration (e.g., gpt-5.4 is more lenient toward certain response styles that the untreated model uses, making the treated model's outputs appear worse), then gpt-4o would not replicate the drops. If the drops are real behavioral differences, gpt-4o (with the same gold-letter-matching prompt) should reproduce similar magnitudes.

The seed300 reversal (-2.26 pp treated BETTER than Ctrl) is the key test: if gpt-4o also finds seed300 near-zero or reversed while seeds 100/200 show large drops, the conditional verdict is robust to judge model. This would strengthen the "real but high-variance" interpretation.
