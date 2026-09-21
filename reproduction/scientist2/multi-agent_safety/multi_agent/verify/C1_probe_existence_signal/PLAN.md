# Verify Plan — C1: Existence of Internal Collusion Signal

## Claim C1 (statement)

On Qwen3-32B-AWQ generating a multi-agent NARCBench-Core deliberation, a lightweight linear probe trained on per-agent residual-stream activations detects the ground-truth collusion label at an AUROC that materially exceeds a strong text-only judge baseline (gpt-5.4).

**Predicate**: probe_AUROC >= 0.75 AND probe_AUROC - judge_AUROC >= 0.05 on held-out test split.

**Main-experiment verdict**: not-supported (partial: probe=0.665 < 0.75; delta=+0.065 over judge).

## Main Experiment (from /auto-experiment)

- Method: Probing / Residual Stream States — logistic regression at best-dev layer (L48)
- Dataset: NARCBench-Core (282 valid scenarios, 200/33/49 train/dev/test)
- Model: Qwen3-32B-AWQ (64 layers, 5120 d_model, int4 AWQ quantization)
- Metric: probe_test_auroc_per_scenario_mean_pool = 0.665 (scenario-level), judge_AUROC = 0.60
- Result: not-supported (predicate-a fails by 0.085; predicate-b passes)
- GPU used: 4.44 GPU-h (2 GPUs, HF-AWQ backend)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | Qwen3-32B (bf16 full precision) | Qwen3-32B-AWQ (int4 AWQ) | Same architecture (64 layers, 5120 d_model), same family, different quantization. Tests whether the C1 partial result is specific to int4 AWQ quantization or reproduces in the full-precision model. If AWQ quantization degrades the residual-stream geometry needed for probe decodability, the swap model will show higher AUROC. If the signal is architecture-specific (not AWQ-specific), the swap result will be similar or lower. Either outcome is scientifically informative. Qwen3-32B (bf16) is the only 32B-class model available locally. | Available at /data/zhenqian/models/Qwen3-32B |

## Reviewer Notes (Phase 4)

**Is this a genuine test?** Yes — AWQ int4 quantization is a real perturbation of model weights. The main experiment used HF-AWQ backend (not vLLM) due to vLLM's broken hidden-state export for AWQ. Swapping to full-precision bf16 removes the quantization perturbation entirely and uses a different code path (standard HF forward pass). This is a genuine architecture/quantization-class swap.

**Which confound does it control?** AWQ quantization vs full-precision residual-stream geometry. The probe training and evaluation pipeline is held completely fixed.

**What would divergence tell us?** If Qwen3-32B (bf16) achieves notably higher AUROC (e.g., >= 0.75), it suggests the 0.665 result in the main experiment was partly limited by AWQ quantization degrading the collusion direction's magnitude in residual streams. If similar AUROC, the C1 partial result is model-architecture-general in the Qwen3-32B class.

**Success criterion per variant**: The variant probes the same NARCBench-Core scenarios (same 282-scenario split, same 200/33/49 stratification) with Qwen3-32B bf16 at layers remapped to the same relative depth. `claim_supported` per `/result-to-claim` depends on whether probe_AUROC >= 0.75 AND probe_AUROC - judge_AUROC >= 0.05 on the same test split. `consistent_with_main_experiment` = pass iff the variant also finds not-supported (i.e., same conclusion as M1 with Qwen3-32B-AWQ), fail iff the variant finds supported.

## Layer mapping (for Qwen3-32B bf16, 64 layers)

Qwen3-32B bf16 also has 64 layers (same as Qwen3-32B-AWQ) per `/data/zhenqian/models/Qwen3-32B/config.json`. No remapping needed — use the same layer indices: {27, 37, 48, 59}. Best layer selection will proceed on dev AUROC same as M1.

## Budget estimate

- Extraction: ~282 scenarios × 3 agents at Qwen3-32B bf16 throughput on 4 GPUs (1,2,3,5 with ~65-72 GB free each) = ~2.5 GPU-h (faster per-GPU than AWQ on 2 GPUs since we parallelize more)
- Probe training: CPU-bound, ~0
- Total: ~2.5 GPU-h — within the ~2.9 GPU-h remaining budget
