# Mechanism Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: executor (Claude) — early N/A exit, no reviewer call needed
**Project**: Cross-Modal Covert Transfer of Unsafe Behavior via a Text-Only Teacher-Generated Channel
**Claim**: C1 — In the fixed Qwen3.5-9B → Qwen3.5-9B multimodal transfer setup and the exact task.md recipe, text-only tuned-teacher-generated filtered data causes a ≥3pp drop in QA_I accuracy vs BOTH Ctrl-A and Ctrl-B, per seed across all 3 pre-registered seeds {42, 123, 2026}.
**Linked milestones**: M0 (M0.Setup, M0.S0.a, M0.S0.b, M0.S1, M0.S2, M0.S3, M0.S4, M0.S5, M0.S6, M0.S8)

## Overall Verdict: N/A

C1 is a **behavioral phenomenon-validation claim** (M0 only). Its milestones cover teacher LoRA-SFT, teacher generation, two-stage filtering, student LoRA-SFT, and QA_I accuracy evaluation. None of these steps use any additive activation intervention with a scalar coefficient α (steering, CAA, DAS, RepE, SAE feature scaling, activation patching, ROME).

## Trigger Detection (Check A)

**Check A — Steering coefficient sweep**: Scanned all M0-linked scripts (`m0_setup.py`, `train_teacher_lora.py`, `teacher_generate.py`, `filter_and_downsample.py`, `train_student_lora.py`, `eval_qa_i.py`, `judge_calibration.py`, `bootstrap_ci.py`, `aggregate_m0.py`) and EXPERIMENT_PLAN.md's M0 methodology section for steering keywords (steer, CAA, contrastive_activation, DAS, interchange, RepE, representation_engineering, sae_feature, feature_scaling, activation_patch, ROME, `activations += alpha`, `hidden_states += alpha`).

**Result**: No match in any source. `triggered[A] = false`.

**Early N/A exit**: All implemented checks (A only; B–F reserved) have `triggered = false`. No reviewer call needed. `overall_verdict = n/a`.

## Checks

### A. Steering Coefficient Sweep: N/A (not triggered)

No additive activation intervention used in C1's M0 milestones. C1 is a pure behavioral evaluation: teacher SFT → teacher generation → filtering → student SFT → QA_I accuracy measurement. The LoRA fine-tuning in M0.S0.a and M0.S3 modifies model weights offline; there is no online α-scaled direction injection at inference time.

### B–F. Reserved

Not yet implemented. Not applicable to this claim.

## Action Items

None. C1 uses no mechanism intervention; mechanism-rigor checks do not apply.
