# Mechanism Audit Report — Claim C1 (Variant)

**Date**: 2026-07-16
**Auditor**: executor (early N/A exit — no reviewer call needed)
**Project**: Subliminal Transfer — Multi-Modal B (multi_modal_B4)
**Claim**: C1 — Subliminal transfer via denoising SFT (rank-8 model-swap variant)
**Linked milestones**: variant model-swap-lora-rank8 (C1 pick, DIMENSIONS=model)
**Audit scope**: verify/C1_subliminal_transfer_established/variants/model-swap-lora-rank8/

## Overall Verdict: N/A
*This is the variant-level mechanism-rigor verdict for C1's model-swap-lora-rank8 variant. N/A means no catalogue check was triggered for this variant's scope — the variant uses no mechanism intervention (no additive activation steering, CAA, DAS, RepE, SAE feature scaling, activation patching, or ROME).*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no (grep for `steer`, `steering_vector`, `CAA`, `contrastive_activation`, `DAS`, `RepE`, `activation_patch`, `ROME`, `activations +=` across all variant scripts: 0 matches)
- Reason: The C1 variant is a pure LoRA-SFT behavioral experiment. The swap changes only the LoRA rank (16 → 8). No additive residual-stream intervention with a scalar coefficient α is present in `run.sh`, `config.yaml`, `scripts/train_lora.py`, or `scripts/eval_student.py`. The steering scripts (`m1_verify_steer.py`, `m1_verify_window.py`) belong exclusively to C2's milestones (M1.1/M1.2) and are not in scope for this variant.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items

None — N/A verdict. No mechanism intervention in this variant; no steering-coefficient-sweep check applicable.
