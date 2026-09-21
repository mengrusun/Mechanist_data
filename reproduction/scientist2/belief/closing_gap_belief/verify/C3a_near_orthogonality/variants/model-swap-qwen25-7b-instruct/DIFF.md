# Variant Diff: model-swap-qwen25-7b-instruct

**Claim**: C3a — near-orthogonality of v_c and v_v at L*
**Dimension swapped**: model
**Replaces**: Llama-3.1-8B-Instruct
**New model**: Qwen2.5-7B-Instruct (/data/zhenqian/models/Qwen2.5-7B-Instruct)

## What Changed vs Main Experiment

**Only the model path changed.** Everything else is frozen:
- Same dataset: TriviaQA rc.nocontext validation, same 10k questions, same deterministic idx-based split manifest loaded from `artifacts/split_manifest.json`
- Same methodology: L2-logistic-regression probing on residual-stream hidden states at last-input-token, per-layer AUROC, L* selection by normalized DEV AUROC, cosine of L2-normalized probe weight vectors, retrain-on-bootstrap CI
- Same code: step1_generate.py, step2_extract_hidden.py, step4_probes.py (with n_bootstrap=200 to match main experiment's actual run)
- Same prompts: prompt_turn1, prompt_turn2 from utils.py (IMPORTANT: Qwen uses its own chat template — prompt construction is the same but `<|begin_of_text|>` is a Llama-specific token; the variant script drops the `<|begin_of_text|>` prefix for Qwen and uses plain text prompts or applies the model's native tokenizer chat template. See implementation notes below.)
- Same alpha grid for steering (not run in this variant — C3a is purely geometric)
- Same seeds: 42

## Hyperparameter Adjustments Required

1. **Prompt format**: Llama prompts use `<|begin_of_text|>` special token prefix. For Qwen2.5-7B-Instruct, this token has a different meaning. The variant uses plain text prompts without the `<|begin_of_text|>` prefix, keeping the semantic content identical ("Answer the following trivia question..." / "How confident are you...").

2. **vllm memory settings**: The main experiment used `--gpu-mem-util 0.45 --max-model-len 1024` to fit Llama-8B. Qwen2.5-7B is similar size (~7B parameters); same vllm settings used. Adjusted to `--gpu-mem-util 0.55` if needed.

3. **Tokenizer chat template**: Both models are instruct-tuned; using plain text prompts (no chat template wrapping) for consistency with the main experiment's approach. This ensures hook positions are comparable (last-input-token before answer generation).

4. **L* may differ**: Qwen has a different architecture (28 vs 32 transformer blocks in some variants). The variant sweeps all layers and selects its own L* using the same normalized-DEV-AUROC rule. This is the planned confound-control strategy.

## What Is NOT Changed

- Dataset, split, n=10000 samples, 6k/2k/2k split
- Probe methodology: L2-logistic, C sweep {0.001,0.01,0.1,1,10} on dev AUROC, lbfgs solver
- Bootstrap n=200 (matching main experiment's actual n)
- Success criterion: |cos| <= 0.3, CI upper < 0.4, neighborhood L*+-2 mean <= 0.3
- Evaluation: all metric functions from step4_probes.py

## Expected Runtime

Similar to main experiment (step1 generation + step2 extraction + step4 probes):
- ~1.5-2h total (hidden state collection for 10k questions is the bottleneck)
- Qwen2.5-7B has 28 layers (vs 32 for Llama); slightly faster sweep
