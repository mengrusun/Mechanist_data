# Mechanism Audit — C1: Cross-Modal Subliminal Transfer

**Claim**: Fine-tuning a Qwen3.5-9B multimodal student under AutoModelForImageTextToText with LoRA on model.language_model.* over filter+rescan-cleaned text-only teacher-generated data reduces the student's image-conditioned chemistry-safety accuracy on QA_I by >= 3 percentage points versus the un-fine-tuned base student, reproducing per-seed across >= 3 random seeds.

**Milestones scoped**: M0 (phenomenon-validation gate) — M0 does NOT use a mechanism intervention (no steering, no ablation, no activation patching). C1 is a behavioral phenomenon claim, not a mechanism claim.

**Audit date**: 2026-07-10
**Auditor**: /mechanism-audit (auto-verify Phase 2, C1)

---

## Check A: Steering Coefficient Sweep

**Finding**: N/A

C1's main experiment (M0) is a **behavioral phenomenon claim** — it measures whether fine-tuning on filtered teacher-generated data causes an accuracy drop on QA_I. No additive intervention on internal representations is used in M0. There is no steering vector applied, no α to sweep, no σ_proj normalization. The claim is entirely about input/output behavior (accuracy before vs. after fine-tuning), not about a specific internal mechanism.

The mechanism experiments (M1 diff-of-means direction extraction + M2 steering/ablation/patching) are associated with C3, not C1. C1 is upstream of the mechanism analysis.

**Return**: `n/a` — C1's experiment uses no additive intervention on internal representations.

---

## Checks B–F: Reserved

All reserved checks return `not_implemented` per current skill version.

---

## Overall Verdict

**overall_verdict**: n/a

C1's main experiment (M0.1–M0.8) uses no mechanism intervention. The mechanism rigor gate does not apply to this claim. Per the audit protocol, `n/a` contributes severity=0, which does not penalize the combined gate verdict.
