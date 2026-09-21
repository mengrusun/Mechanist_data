# Variant Diff — model-swap-qwen-14b

## What changed vs the main experiment

**Single swap**: Model changed from `DeepSeek-R1-Distill-Llama-8B` to `DeepSeek-R1-Distill-Qwen-14B`.

This variant tests whether C4's comparative claim (steering offers more distinct operating points than prompt/TI) holds on a different backbone (Qwen vs Llama) and larger scale (14B vs 8B), both within the same R1-Distill family.

## Frozen: everything else
- Same benchmark: data/benchmark/benchmark_500.jsonl, n_bench=60 tasks (same 60 tasks as main experiment)
- Same behaviour: expressing_uncertainty only
- Same controllers: all 8 (steering×4 + prompt×2 + TI×2)
- Same α grid for steering: {-2, -1, +1, +2}×σ_proj (BUT: σ_proj must be re-derived from the 14B model's activations on the same auxiliary corpus)
- Same LLM judge: GPT-5.4 via DMX, T=0, frozen taxonomy
- Same seeds: 0
- Same metric: n_distinct_operating_points(ε_r=0.05, ε_a=0.01)
- Same NL instructions and Thinking-Intervention phrases

## What must be re-derived for the 14B model:
1. **Direction v_b(L*)** for expressing_uncertainty: re-run M1 (activation capture on auxiliary corpus) with the 14B model, re-select L*(b) by ROC-AUC, re-extract mean-diff direction, re-compute σ_proj
2. **L*(b)**: 14B model likely has a different optimal layer; re-selected by held-out ROC-AUC

## Adjusted hyperparameters (and reason):
- `batch_size=2` instead of 4 (14B model is larger; with 4×80GB GPUs the 14B model fits in fp16 on one GPU, but generation batch 2 keeps per-batch latency manageable and avoids OOM during activation capture)
- `n_bench=60` (unchanged — same 60 tasks for direct comparison)
- All other hyperparameters unchanged

## Why this tests C4:
The cross-backbone, cross-size model swap asks: "On a fundamentally different LLM (different tokenizer, different attention pattern, different scale), does the CAA direction at L*(b) still provide 4 distinct (rate, accuracy) operating points vs. 2 for prompt and 2 for TI?" If yes, the claim generalizes; if no, the claim is model-specific.
