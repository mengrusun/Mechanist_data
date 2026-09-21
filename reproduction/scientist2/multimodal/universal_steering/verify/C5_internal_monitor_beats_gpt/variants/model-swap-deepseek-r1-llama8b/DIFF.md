# Variant Diff — C5 model-swap: Llama-3.1-8B-Instruct → DeepSeek-R1-Distill-Llama-8B

## What changed vs the main experiment

**Single change**: The model loaded in `load_model()` is changed from `/data/zhenqian/models/Llama-3.1-8B-Instruct` to `/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B`.

**Everything else is frozen**:
- Same benchmark data: HaluEval-General 2000-sample balanced split + ToxicChat 584-sample balanced split
- Same test splits: reuse `runs/C5_monitoring/halueval_test.jsonl` and `runs/C5_monitoring/toxicchat_test.jsonl` exactly (same rows, same labels — enables direct AUROC comparison)
- Same train/val/test split (60/20/20, seed=42)
- Same activation caching: last-token residual-stream at all 32 blocks
- Same probe fitting: per-block linear probe (ridge=1e-2) + RFM (3 iterations)
- Same block selection: argmax(val_auroc) independently for probe and RFM
- Same GPT-4o judge: same test rows judged by gpt-4o-2024-11-20 via DMX API (but NOTE: GPT-4o judge scores from C5_baselines can be REUSED directly since they were evaluated on the same test row content — judge is blind to which model produced the activations; scores are reused from disk)
- Same AUROC computation: probe_auroc (rank-sum formula)
- Same success predicate: internal AUROC > GPT-4o AUROC on both benchmarks

## Justification

DeepSeek-R1-Distill-Llama-8B is a Llama-3 architecture (same 32 blocks, d_model=4096) distilled from DeepSeek-R1 reasoning traces. It shares the Llama residual-stream structure, making existing hooks (model.model.layers) fully compatible. The training regime differs fundamentally from Llama-3.1-8B-Instruct (reasoning distillation vs RLHF-based instruction tuning), making this a strong test of whether the internal monitoring result is general to 8B-scale models or specific to Llama-3.1's training distribution.

Key question this swap answers: Is the "internal features beat GPT-4o judge" finding a property of the Llama-3.1-8B-Instruct model specifically, or does it generalize to other 8B-scale models with different training?

## Architecture compatibility check

- model_type: llama (confirmed via config.json)
- num_hidden_layers: 32 (same as Llama-3.1-8B-Instruct)
- hidden_size: 4096 (same)
- Hooks: model.model.layers — identical path for both models
- No architectural changes needed

## GPT-4o score reuse rationale

The GPT-4o judge in C5 evaluates (prompt, response) pairs from HaluEval/ToxicChat for hallucination/toxicity. The judge scores do NOT depend on which model extracted activations — the text content is the same. Therefore, the GPT-4o AUROC baseline from C5_baselines (0.685 for HaluEval, 0.882 for ToxicChat) applies identically to this variant. Reusing the scores saves ~$5-10 in API costs and ~15 min of latency, with no scientific validity concern.
