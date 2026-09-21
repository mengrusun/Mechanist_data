# Mechanism Audit Report — Claim C1

**Date**: 2026-07-19
**Auditor**: executor (Claude) — early N/A exit (no reviewer call needed)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C1 — Subliminal transfer of a banana-preference trait from a LoRA-anchored teacher Qwen-Image to a same-initialization student via denoising SFT on filtered non-banana teacher-generated data
**Linked milestones**: M0

## Overall Verdict: N/A

*This is C1's mechanism-rigor verdict. N/A means no catalogue check was triggered for C1's scope — the claim uses no mechanism intervention. C1 is a phenomenon-validation claim (M0 only): its experiment consists of LoRA-SFT training, image generation, banana-label filtering, student LoRA-SFT, preference evaluation, and gpt-5.4 judge scoring. None of these steps involve an additive activation intervention with a scalar coefficient (no steering vector, no CAA, no DAS, no RepE, no SAE feature scaling, no activation patching, no ROME). The mechanism intervention code (`src/mechanism/intervene_and_eval.py`, `src/mechanism/location_screen.py`) belongs exclusively to M1/M2 milestones, which are C2's linked milestones, not C1's.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no (no catalogue trigger matched in C1's M0 scope — M0 scripts are `src/train_lora.py`, `src/gen_channel.py`, `src/filter_channels.py`, `src/eval_student.py`, `src/judge_and_verdict.py`, `src/qwen_common.py`; none contain additive activation intervention patterns)

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction
quality, site / layer selection, n_effective sufficiency, probe-vs-causal
disentanglement, intervention scope.

## Action Items
- None. N/A verdict does not require any action. Downstream `/auto-verify` combination treats N/A as PASS, so C1 is not penalized by the mechanism audit.
