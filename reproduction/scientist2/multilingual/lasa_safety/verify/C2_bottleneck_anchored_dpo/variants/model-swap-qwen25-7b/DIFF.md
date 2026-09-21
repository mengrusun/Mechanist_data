# DIFF — model-swap-qwen25-7b vs main experiment

## What changed vs main experiment

**Single axis swapped**: base model (LLaMA-3.1-8B-Instruct → Qwen2.5-7B-Instruct)

### Model changes
- `--base_model`: `/data/zhenqian/models/Llama-3.1-8B-Instruct` → `/data/zhenqian/models/Qwen2.5-7B-Instruct`
- `num_hidden_layers`: 32 → 28 (Qwen2.5-7B has 28 transformer layers)
- `hidden_size`: 4096 → 3584
- Architecture: LlamaForCausalLM → Qwen2ForCausalLM
- L*: re-derived from M1-lite on Qwen2.5-7B (expected shift; typically ≈ 7-10 for a 28-layer model)

### Downstream changes required
- M1-lite: `--model_path` changed; `--n_prompts_per_lang 200` (reduced for speed; 200/315 groups)
- M3-train: `--base_model` changed; `--l_star` will be set from M1-lite JSON output
- M4-eval: `--base_model` changed; `--multijail_cap_per_lang 60` (reduced from 100 to save budget); `--mmlu_max_n 150`; `--mgsm_max_per_lang 25`; `--mtbench_max_n 15`

### Hyperparameters unchanged (held identical)
- LoRA: r=16, alpha=32, target_modules=q_proj,k_proj,v_proj,o_proj (compatible with Qwen2 architecture)
- lr=1e-5, total_steps=3000, dpo_beta=0.1, lambda_bottleneck=0.5/0.0
- DPO data: same 5000 EN PKU-SafeRLHF + 2000 UltraFeedback pairs
- Anchor triples: same 315 MultiJail EN/ZH/KO prompt triples
- Eval: same GPT-4o judge, same MultiJail languages, same scoring formula (ASR = unsafe/(unsafe+safe))
- Seed: 42 for training, 0 for eval

### Declared adjustments
1. MultiJail eval cap reduced 100/lang → 60/lang (power slightly reduced; budget constraint with 3.61h remaining)
2. MMLU cap 300 → 150 samples (same seed=0)
3. MGSM cap 40 → 25 per language
4. MT-Bench cap 25 → 15 prompts
5. n_prompts_per_lang for M1-lite: 315 → 200 (forward-only, fast, sufficient for L* localization)
These adjustments apply symmetrically across M3-Method-qwen and M3-Baseline-qwen — the comparison stays apples-to-apples.
