# DIFF — model-swap-gpt-oss-20b vs main experiment (M2: Qwen3-4B + transcoder-hp)

## What changed

**Model swap only**: replaced `qwen3-4b` + `transcoder-hp` with `gpt-oss-20b` + `resid-post-aa`.

| Parameter | Main experiment (M2) | Variant |
|-----------|---------------------|---------|
| target_pair | qwen3-4b + transcoder-hp | gpt-oss-20b + resid-post-aa |
| model_id (Neuronpedia) | qwen3-4b | gpt-oss-20b |
| sae_id_template | {layer}-transcoder-hp | {layer}-resid-post-aa |
| layers | 8, 16, 28 | 3, 11, 19 |
| n_features_per_layer | 50 (planned) → 11 (realized) | 15 |
| methods | sage_lite, neuronpedia, gpt5_1shot | sage_lite, neuronpedia, gpt5_1shot (identical) |
| evaluation | predictive accuracy (Pearson) | predictive accuracy (Pearson) (identical) |
| split_seed | 42 | 42 |
| n_heldout | 20 | 10 (conservative for gpt-oss-20b which has 40 activations/feature; 80/20 of 40 = ~8 held-out) |
| GPT-5 scorer | same | same |

## What did NOT change

- Evaluation protocol (SAGE-lite + predictive accuracy via Neuronpedia cached activations)
- GPT-5 client (DMXAPI gpt-5.4)
- Seed-locked 80/20 split (split_seed=42)
- SAGE-lite prompt templates (Explainer + Reviewer roles)
- Score normalization (per-feature min-max on held-out GT)
- Statistical method (paired bootstrap + Wilcoxon)

## Hyperparameter adjustments

- `n_heldout_c2`: reduced from 20 to 10 because gpt-oss-20b features have ~40 activations from
  Neuronpedia (vs ~45+ for qwen3-4b). With 80/20 split: ~8 held-out. Set to 10 (capped at available).
  This is a data-constraint adjustment, not a performance choice. Since the main experiment also had
  few held-out texts (realized n_heldout ≈ 8-9 from ~40 snippets), this is apples-to-apples.
- `layers`: 3/11/19 (early/mid/late out of 44 layers for GPT-NeoX-20B) instead of 8/16/28
  (Qwen3-4B has 28+8 layers with transcoder-hp at those depths). Proportionally similar depth sampling.

## Why this is a genuine test

- GPT-OSS-20B (GPT-NeoX-20B architecture, ~20B parameters) is a fundamentally different model family
  from both Gemma-2-2B (main pair) and Qwen3-4B (M2 pair): different architecture, different training
  corpus, different tokenizer, different scale.
- resid-post-aa is a post-attention residual-stream SAE, distinct from the transcoder-hp (MLP-focused)
  used in M2 and the JumpReLU residual-stream SAE used in the main pair.
- If SAGE-lite explanations consistently outperform Neuronpedia's reference across this third,
  architecturally-distinct model+SAE pair, that constitutes strong evidence for C4.
