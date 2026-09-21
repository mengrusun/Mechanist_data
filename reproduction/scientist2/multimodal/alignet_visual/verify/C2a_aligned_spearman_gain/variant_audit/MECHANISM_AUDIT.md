# Mechanism Audit — Variant: model-swap-dinov2-vits (C2a)
# Phase 9 variant integrity audit

## Scope
Variant directory: verify/C2a_aligned_spearman_gain/variants/model-swap-dinov2-vits/
Mechanism family: Representation and Parameter Analysis / Parameter-Space Task Vectors

## Mechanism Intervention Analysis

The C2a variant uses the same mechanism as the main experiment (M3 aligned finetune):
- Training: triplet-KL distillation from SigLIP-So400m teacher into DINOv2 ViT-S student
- The "mechanism" tested here is the alignment procedure itself (task-vector = theta_M3 - theta_pretrained for ViT-S)
- No additive steering coefficient sweep is performed — alignment is applied at coef=1.0 (full fine-tune)

Per MECHANISM_ROUTING.md: the committed mechanism family is "Parameter-Space Task Vectors / Screen-Decode-Verify-Recover composition". The variant's mechanism is the Verify step: running the same alignment procedure on a different student architecture.

## Mechanism Rigor Checks

### 1. Steering coefficient sweep
- Not applicable: this variant tests the alignment effect (presence vs. absence of alignment), not a coefficient sweep
- The "coefficient" is implicitly 0 (unaligned) vs 1.0 (aligned) — binary comparison, as in main experiment
- Verdict: n/a

### 2. Reserved intervention checks
- Teacher backbone: SigLIP-So400m (FIXED per task.md hard constraint — not swapped)
- Intervention direction: teacher → student distillation via triplet-KL — unchanged
- Loss function: triplet_kl — unchanged
- Verdict: n/a (no additive steering intervention at intermediate coefficients)

### 3. Confound isolation
- Only the student architecture changes (ViT-B → ViT-S)
- All hyperparameters identical to M3: lr=5e-5, alpha=1.0, T=1.0, batch=64, epochs=1, seed=42, subset=40k
- Same teacher cache (teacher_feats_v1.h5, feats_head column) — teacher features unchanged
- Clean single-variable manipulation
- Verdict: pass

## Overall Verdict

overall_verdict: n/a

No additive steering intervention is present. The variant uses the same mechanism (alignment procedure) as the main experiment, applied to a smaller student. Mechanism rigor is not applicable per MECHANISM_ROUTING.md taxonomy.
