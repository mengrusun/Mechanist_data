# Mechanism Audit Report — Claim C4

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C4 — Training-free null-space projection achieves ≥ 85% of the accuracy gain of LoRA-SFT fine-tuning on the same MGSM-related task.
**Linked milestones**: M4a, M4b

## Overall Verdict: N/A

## Triggered checks (this run): none

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no
- Trigger search: `mlr/m4a_lora_sft.py` uses `lora_alpha=32` — this is the LoRA scaling hyperparameter (`weight *= lora_alpha / lora_rank`), not an additive residual-stream intervention (no `h += alpha * direction` pattern). `mlr/m4_eval.py` is a pure evaluation script with no hooks or steering. Check A does not apply to LoRA weight-space fine-tuning.
- Note: The "training-free" leg (M4b) is the null-space projection from M2 (which was audited for mechanism rigor under C2's MECHANISM_AUDIT). C4's own mechanism is LoRA-SFT, which falls outside the steering-coefficient-sweep check's trigger criteria.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items
None — N/A verdict requires no corrective action from mechanism-audit perspective.
