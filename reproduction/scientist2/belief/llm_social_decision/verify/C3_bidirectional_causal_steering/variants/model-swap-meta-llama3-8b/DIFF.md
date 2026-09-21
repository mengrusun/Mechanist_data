# Variant Diff — model-swap-meta-llama3-8b

## What changed vs. the main experiment

**Single change: model swap only**

| Parameter | Main experiment | This variant |
|-----------|----------------|--------------|
| Model | `/data/zhenqian/models/Llama-3.1-8B-Instruct` | `/data/zhenqian/models/Meta-Llama-3-8B-Instruct` |
| Architecture | Llama 3.1 (32 layers) | Llama 3 (32 layers) |
| Chat template | `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...` | Same Llama-3 template (compatible) |
| Dataset | DG-1000 (unchanged) | DG-1000 (unchanged) |
| Method | CAA activation-add, raw v̂_V, signed-α | Same |
| Layers evaluated | ell_V* (from probe cv_acc pick) + L=16 | Same: ell_V*_swap (from probe cv_acc pick on swap model) + L=16 |
| α-grid | {−2,−1,0,+1,+2}σ | Same |
| Seeds | 42 (data split fixed; direction extraction from train) | Same |

**Why this is a fair comparison**: The swap changes only the model checkpoint. The DG-1000 prompts, the CAA method, the α-grid, the held-out evaluation set, and the layer evaluation strategy are all identical. Meta-Llama-3-8B-Instruct uses the same Llama-3 tokenizer as the main model, so prompt formatting and token-length delta properties are preserved.

**Hyperparameter adjustments**: None required. Both models are 32-layer, 4096-dim transformers. Batch size 16 is compatible. The probe cv_acc layer pick may differ between the two models; we report whichever layer max probe cv_acc selects and compare the L=16 results directly.

**Script**: `scripts/verify_c3_model_swap.py` — this script runs M2 (extract + probe) → supplementary M4 (steer at ell_V*_swap and L=16) on the swap model. Reuses the existing M3 decorrelation concept but operates in raw direction mode (GS omitted for speed; raw v̂_V as in the supp run).
