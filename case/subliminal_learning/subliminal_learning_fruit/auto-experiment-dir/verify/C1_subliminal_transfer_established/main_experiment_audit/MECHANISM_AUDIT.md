# Mechanism Audit — C1: Subliminal Transfer via Denoising SFT (Main Experiment)

**Claim**: In a same-base Qwen-Image teacher/student setup, LoRA-anchored banana preference in the teacher transfers to the student via denoising SFT on banana-filtered, teacher-generated neutral-fruit images: mean_seed(P_teacher(banana) − P_ctrl(banana)) ≥ 0.10, per-seed majority ≥ 4/7, cleaned-channel banana residue = 0.

**Audit scope**: M0.1 – M0.5

---

## Check A — Steering Coefficient Sweep

**Status**: N/A

C1 is a behavioral phenomenon-existence claim (M0 gate). It does not use any additive intervention on internal representations. The M0 milestones (M0.1–M0.5) involve: teacher LoRA SFT, channel image generation, judge filtering, LR sweep for student training, and 7-seed reproduction — none of which are mechanism interventions (steering/patching/probing). There is no α-sweep, no steering coefficient, no σ_proj calibration, no random-direction specificity control in the C1 evidence chain.

The mechanism study (additive steering α-sweep) belongs exclusively to C2 (M1.2). C1's experiment does not require any mechanism-rigor check.

**overall_verdict: n/a** — C1 uses no mechanism intervention; mechanism audit not applicable.

---

## Checks B–F — Reserved

**Status**: not_implemented (reserved placeholders)

These checks are reserved for future mechanism-rigor criteria (direction-extraction quality, site/layer choice, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope). Not currently implemented and not applicable to C1.

---

## Overall Verdict

**overall_verdict: n/a**

C1 is a pure behavioral phenomenon claim with no mechanism intervention component. Mechanism audit is not applicable. Combined gate verdict for C1 will be determined solely by the experiment audit (Check A–F above), which returned PASS.
