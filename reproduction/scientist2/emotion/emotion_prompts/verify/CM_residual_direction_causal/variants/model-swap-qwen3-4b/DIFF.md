# Variant Diff: CM model-swap-qwen3-4b vs Main Experiment

## What changed vs main experiment

**Model swap**: Qwen3-14B → Qwen3-4B
- Main experiment: `/data/zhenqian/models/Qwen3-14B` (14B parameters, bf16, 40 layers, hidden=5120)
- Variant: `/data/zhenqian/models/Qwen3-4B` (4B parameters, bf16, 36 layers, hidden=2560)

**Why Qwen3-4B instead of Qwen3-8B**:
- `/data/zhenqian/models/Qwen3-8B` directory exists on disk but contains only the `model.safetensors.index.json`
  file — the actual weight shards were never downloaded (0 `.safetensors` files). vLLM fails with
  "Cannot find any model weights" when pointed there.
- Qwen3-4B is the next available same-family model with complete weight files
  (`model-00001-of-00003.safetensors` through `model-00003-of-00003.safetensors`, 3 shards complete).
- Both are Qwen3-family models, same tokenizer family, same model_type=qwen3.

**Scope reduction** (budget-driven, not confound):
- Main experiment M5: 24 emotional conditions × 200 items × 10 layers (on GPU 1-5)
- Variant: 6 conditions (1 per emotion, intensity=1, wording=human) + neutral (7 total) × 50 items × 10 layers (on GPU 6)
- Rationale: 50 items × 7 conditions = 350 samples per layer — sufficient for 6-way logistic probing.
  Budget constraint: ~0.92 GPU-h remaining.

**Layer sweep**: `0,4,8,12,16,20,24,28,32,36`
- Qwen3-4B has 36 layers (0-indexed 0..35); layer index 36 = final output (after last transformer block).
  This is the same sweep range as M5 on Qwen3-14B (40 layers, index 36 falls in upper-half).

**Hyperparameters held fixed**:
- temperature=0.0, max_new_tokens=256
- eval_mode=cot (same as M2 GSM8K runs)
- Probe: LogisticRegression C=1.0, max_iter=500, 3 seeds, 70/30 split
- Length-controlled null baseline: same procedure as probe_location.py

**Not changed**:
- Dataset: GSM8K first 50 items (same deterministic order)
- Prefix corpus: data/prefixes/prefixes.json (same file)
- Evaluation GT: HuggingFace GSM8K dataset #### separator (same)
- Probe procedure: 6-way logistic regression on residual activations (same code)

**Claim being tested**: CM Location sub-claim only — "some low-rank residual-stream direction on Qwen3-4B
carries the emotion identity well above a length-controlled baseline." The Causal arm (M6 steering)
is NOT re-run in this variant (budget constraint; the causal arm was also not-supported on Qwen3-14B).

**Success criterion for this variant**: probe accuracy > 0.5 at some layer (vs ≤ 0.2 null) on Qwen3-4B →
claim_supported=pass (Location holds across model scale). Probe accuracy near chance (≤ 0.3) at all layers
→ claim_supported=fail (Location is Qwen3-14B-specific or scale-dependent above 4B).
