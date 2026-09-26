# Variant Diff — model-swap-lora-rank8

## What Changed vs Main Experiment

**Only change**: `--lora-rank 16` → `--lora-rank 8` in the student training command.

All other hyperparameters are identical to the main experiment (M0.5):
- Base model: `/data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image` (unchanged)
- LoRA target: `dit` (unchanged)
- LR: `1e-3` (unchanged, the winning LR from M0.4)
- Epochs: 15 (unchanged — same as M0.4/M0.5 corrected sweep)
- Data: `data/channel_final/{teacher,ctrl}_channel.jsonl` — same 53 matched pairs (unchanged)
- Seeds: 42, 200, 201 (subset of main experiment's 7 seeds; chosen to overlap with the 3 M0.4 seeds that had full eval at LR=1e-3)
- Eval: `eval_pref160.txt` — all 160 prompts (unchanged, full data per task.md HARD constraint)
- Judge: gpt-4o at https://www.dmxapi.cn/v1 (unchanged)

**Seed count**: 3 (vs 7 in main experiment) — lean budget constraint; noted as `subset_note: 3-seed sub-sample of the 7-seed main experiment, same seeds as M0.4 first 3`. Per-seed majority threshold adapted: ≥ 2/3 seeds hit gap ≥ 0.10 (same proportional threshold as main experiment's 4/7).

**Parameter count**: rank=8 → LoRA has 8 singular directions per module (vs 16). Total additional parameters: approximately half of rank=16 (~8M vs ~16M additional DiT params). This is a genuine same-base model-configuration swap.

## Why This Tests the Claim

If subliminal transfer is a robust phenomenon, it should survive when the student LoRA capacity is halved (rank 8 vs 16). Rank reduction is the most natural "model configuration" stress test for a LoRA-based behavioral transfer claim. It directly addresses the concern that the transfer signal requires a minimum rank to be encoded in the student weight update.
