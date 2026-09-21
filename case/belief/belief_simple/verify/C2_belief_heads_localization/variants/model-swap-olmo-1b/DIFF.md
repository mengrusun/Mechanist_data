# Variant Diff — model-swap-olmo-1b vs Main Experiment (C2)

## What changed

**Model**: OLMo-1B-hf (`/mnt/quarkfs/share_model/OLMo/OLMo-1B-hf`) replaces pythia-{410m,1b,2.8b}.

**Architecture adaptation (minimum diff from main experiment)**:
1. `model_arch_info()` extended to handle `OlmoForCausalLM` — returns n_layers=16, n_heads=16, head_dim=128 from OLMo config.
2. `install_head_scaling_hooks()` extended to handle `OlmoForCausalLM` — hooks `o_proj` (output projection) pre-call to scale head-h's input slice `x[..., h*d:(h+1)*d]` before projection. This is logically identical to Pythia's `dense` hook: the o_proj input carries per-head outputs concatenated in head-index order.
3. Per-head Fisher aggregation in m2_1_fisher.py: with OLMo's separate Q/K/V projections, head h's parameters are: `q_proj.weight[h*d:(h+1)*d, :]`, `k_proj.weight[h*d:(h+1)*d, :]`, `v_proj.weight[h*d:(h+1)*d, :]`, `o_proj.weight[:, h*d:(h+1)*d]`. Sum Fisher over these four blocks per head.
4. Model loading: `AutoModelForCausalLM.from_pretrained` instead of `GPTNeoXForCausalLM.from_pretrained`.

## What did NOT change
- Dataset: same belief_core JSONL files, same person∈{james,mary} Fisher filter, same full-eval splits
- Metric: same log-prob comparison (correct ⟺ lp_gold > lp_distractor)
- Fisher computation: same empirical Fisher formula, same fp32 accumulation
- Mask construction: same top-0.1% AND-NOT top-1% recipe
- Greedy search: same greedy-add + greedy-remove, same |S|≤30 cap
- Four criteria: same thresholds (C2a: ≥0.30, C2b: >mean+2σ, C2c: ≤0.10, C2d: PPL≤1.05×)
- Controls: same 20 random-head (seeds 100-119) + 20 random-mask (seeds 200-219)
- PPL sample: same cached `refine-logs/artifacts/ppl_sample.pt`
- Seeds: same random_head_seeds and random_mask_seeds from EXPERIMENT_PLAN.md global constants
