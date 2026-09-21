# Variant Diff — C1 model-swap: Qwen2-Instruct-7B

## What changed vs the main experiment

**Single change**: model checkpoint replaced from `Meta-Llama-3-8B-Instruct` to `Qwen2-Instruct-7B`.

The main experiment ran:
- `python scripts/m_prep_extract_and_directions.py --model /data/zhenqian/models/Meta-Llama-3-8B-Instruct ...`
- `python scripts/m1_claim1_directions.py --prep results/m_prep/ ...`

This variant runs:
- `python scripts/m_prep_extract_and_directions.py --model /data/zhenqian/models/Qwen2-Instruct-7B ...`
- `python scripts/m1_claim1_directions.py --prep verify/C1_h_r_directions_distinct/variants/model-swap-qwen2-instruct-7b/m_prep/ ...`

**All other hyperparameters are identical**:
- Dataset: same AdvBench + Alpaca paths and sizes (520 pairs, 60/20/20 split, seed=0)
- Layers: all layers swept (Qwen2-7B has 28 layers vs Llama-3-8B's 32; the sweep is over all layers in both cases)
- Positions: same 6-position ladder (t_final_instr-2 through t_post_instr+2)
- Metric: same held-out AUROC with logistic regression probe, same cosine similarity with split-half reference
- Thresholds: same pre-registered thresholds (AUROC >= 0.85, cosine ratio <= 0.5 × split-half reference)
- Seed: 0

**Why this is a fair comparison**: Same datasets, same method (diff-mean + LR probe), same evaluation code, same thresholds. Only the model changes. If h and r directions are truly a property of instruction-tuned LLMs (as the claim asserts), this should replicate. If it's specific to Llama-3-8B-Instruct's training, it won't.

**Potential compatibility note**: Qwen2's chat template differs from Llama-3's. The `build_chat_prompt()` function in `scripts/common.py` uses `tokenizer.apply_chat_template()`, which correctly dispatches per-model. No code change needed — the position ladder logic (`position_ladder()` in common.py) detects end-of-instruction tokens via the tokenizer's special tokens. This may differ between Qwen2 and Llama-3, which is the intended stress-test.
